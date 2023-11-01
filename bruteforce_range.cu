#include <iostream>
#include <chrono>
#include <array>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <cassert>
#include "bruteforce_range.cuh"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Count how many bits of the suffix are "varying".
// For example, for :
// a=10010001
// b=10010111
// the common prefix is [10010] and the suffix is [001] or [111] - 3 bits
uint64_t variable_suffix_bits(uint64_t a, uint64_t b)
{
    uint64_t suffix_len = 0;
    while (a != b)
    {
        suffix_len += 1;
        a >>= 1;
        b >>= 1;
    }
    return suffix_len;
}

uint32_t variable_suffix_bits_optimized(uint32_t a, uint32_t b)
{
    uint32_t suffix_len = 0;
    while (a != b)
    {
        suffix_len += 1;
        a >>= 1;
        b >>= 1;
    }
    return suffix_len;
}

// Determines the value of key[0] based off the input simulating
// the output of the first call to QueryPerformanceCounter()
// and the maximum number of ticks we expect the second call to have returned
// as defined in MAX_PERFCOUNTER_SECOND_CALL_TICKS_DIFF
// This is XOR'd with what the input was to simulate the call GetTickCount()
void BruteforceRange::perfcounter_xor_set()
{
    uint64_t min_pc = perfcounter_.get();
    uint64_t max_pc = perfcounter_.get() + MAX_PERFCOUNTER_SECOND_CALL_TICKS_DIFF;

    uint64_t min_gtc = gettickcount_.min();
    uint64_t max_gtc = gettickcount_.max();

    uint64_t gtc_bits = variable_suffix_bits(min_gtc, max_gtc);

    uint64_t pc_step = uint64_t{1} << gtc_bits;
    uint64_t pc_mask = ~(pc_step - uint64_t{1});

    uint64_t gtc_prefix = (min_gtc & pc_mask);
    uint64_t min_pc2 = min_pc ^ gtc_prefix;
    uint64_t max_pc2 = max_pc ^ gtc_prefix;
    if (max_pc2 < min_pc2)
    {
        std::swap(min_pc2, max_pc2);
    }

    uint64_t min_pc3 = (min_pc2 & pc_mask);
    uint64_t max_pc3 = (max_pc2 & pc_mask) + pc_step;

    perfcounter_xor_ = BruteforceParam<uint64_t>(min_pc3, max_pc3);
}

// Moves the internal state one step forward. Returns false if the range is done.
bool BruteforceRange::forward()
{
    // If we ran out of keys return false right away
    // We need this because our BruteforceParam will wrap around when asking for the next key
    // and when we call the function to rotate the keys, we don't exit on first failed attempt
    // as to allow all packets currently being processed to finish
    if (no_more_keys_)
    {
        return false;
    }

    if (perfcounter_xor_.next())
    {
        // We don't increment cur_keyspace_index_ here xor because we don't calculate the total keyspace with the xor iterations
        // See keyspace_ in the constructor for more info
        return true;
    }

    if (perfcounter_.next())
    {
        perfcounter_xor_set();
        cur_keyspace_index_++;
        return true;
    }
    perfcounter_xor_set();

    if (filetime_.next())
    {
        cur_keyspace_index_++;
        return true;
    }

    // If we only have 1 TID in the vector that means its our last TID and theres no more to check
    // If we have more than 1 then we remove the last element since we use tids_.back() for key[4]
    if (tids_.size() > 1)
    {
        tids_.pop_back();
        cur_keyspace_index_++;
        return true;
    }

    no_more_keys_ = true;
    return false;
}

BruteforceRange::BruteforceRange(uint32_t pid, std::vector<uint32_t> tids, BruteforceParam<uint32_t> gettickcount,
                                 BruteforceParam<uint64_t> filetime, BruteforceParam<uint64_t> perfcounter)
    : pid_(pid), tids_(tids), gettickcount_(gettickcount), filetime_(filetime), perfcounter_(perfcounter), perfcounter_xor_(0, 0),
      cur_keyspace_index_(0)
{ // hardcoded perfcount_diff

    // We calculate the keyspace not accounting for the second call to QueryPerformanceCounter()
    // not because we shouldn't but because the person writing this comment wasn't sure how and
    // felt time was better spent optimizing than figuring out that math. If you know how, please do.
    keyspace_ = tids_.size() * filetime_.keyspace() * perfcounter_.keyspace();

    // Assume we start at the beginning
    start_when_ = 0;

    // Assume we end at the end
    // We subtract 1 because keyspace_ is the total size but arrays index start at 0
    // and end at one less than the total count
    done_when_ = keyspace_ - 1;

    // We are just starting so there should be keys to check
    no_more_keys_ = false;

    // We start at the beginning but may end up skipping some when we apply start_when_ logic
    cur_keyspace_index_ = 0;

    perfcounter_xor_set();
}

// Returns false if the range is done, otherwise sets new key in packet.
bool BruteforceRange::next(Packet target, PacketStatus *status, int thread_id)
{

    // If we aren't at the starting key, keep moving forward until we are
    while (cur_keyspace_index_ < start_when_)
    {
        forward();
    }

    // If the user requested we stop early do so
    if (cur_keyspace_index_ > done_when_)
    {
        *status = PacketStatus::KeySpaceExhaustedOrKeyFound;
        return false;
    }

    // If we ran out of keys to try
    if (!forward())
    {
        *status = PacketStatus::KeySpaceExhaustedOrKeyFound;
        return false;
    }

    // Prepare the packet with a new key to be SHA'd
    uint32_t key[8];
    //                                                // These comments represent what the data was in the original Phobos
    key[0] = perfcounter_xor_.get();                  // Second call to QueryPerformanceCounter() xor'd against GetTickCount()
    key[1] = (perfcounter_.get() >> 32) & 0xFFFFFFFF; // First call to QueryPerformanceCounter()
    key[2] = perfcounter_.get() & 0xFFFFFFFF;         // First call to QueryPerformanceCounter()
    key[3] = pid_;                                    // GetCurrentProcessId()
    key[4] = tids_.back();                            // GetCurrentThreadId()
    key[5] = (filetime_.get() >> 32) & 0xFFFFFFFF;    // GetLocalTime() + SystemTimeToFileTime()
    key[6] = filetime_.get() & 0xFFFFFFFF;            // GetLocalTime() + SystemTimeToFileTime()
    key[7] = (perfcounter_.get() >> 32) & 0xFFFFFFFF; // Second call to QueryPerformanceCounter(). I think the idea behind this is that sinces its the high bits, there is a "high probability" that the first call and second call to QueryPerformanceCounter() will have the same high bits // with high probability

    memcpy(target, key, 32);

    // Mark packet as ready to be SHA'd
    *status = PacketStatus::ReadyForSHA;

    // We have successfully generated a new key so increment our counter and return true
    total_keys_tried_++;
    return true;
}

// What key iteration are we currently on?
uint64_t BruteforceRange::current() const { return cur_keyspace_index_; }

// How many keys are there to try?
uint64_t BruteforceRange::keyspace() const { return keyspace_; }

// When are we going to stop trying? (The user can ask to not try the whole keyspace)
uint64_t BruteforceRange::done_when() const { return done_when_; }

// How many keys have we tried? This will not count those we skipped over due to the user asking us to not start at the beginning
uint64_t BruteforceRange::total_keys_tried() const { return total_keys_tried_; }

// How many iterations have we returned for trying
// Expressed as a float from 0-1
float BruteforceRange::progress() const
{
    return static_cast<float>(cur_keyspace_index_) / done_when_;
}

BruteforceRange BruteforceRange::parse(std::string path)
{
    std::ifstream i(path);
    if (!i.is_open())
    {
        std::cout << "Failed to open file: " << path << std::endl;
        // Handle the error, maybe throw an exception or return a default object
    }

    json j;
    i >> j;
    std::cout << "Successfully read JSON from file: " << path << std::endl;

    uint32_t pid = j["pid"];
    std::cout << "Parsed PID: " << pid << std::endl;

    std::vector<uint32_t> tids = j["tid"];
    std::cout << "Parsed TIDs count: " << tids.size() << std::endl;

    json g = j["gettickcount"];
    json p = j["perfcounter"];
    json f = j["filetime"];

    std::cout << "Initializing gettickcount with min: " << g["min"] << ", max: " << g["max"] << ", step: " << g["step"] << std::endl;
    BruteforceParam<uint32_t> gettickcount(g["min"], g["max"], g["step"]);

    std::cout << "Initializing filetime with min: " << f["min"] << ", max: " << f["max"] << ", step: " << f["step"] << std::endl;
    BruteforceParam<uint64_t> filetime(f["min"], f["max"], f["step"]);

    std::cout << "Initializing perfcounter with min: " << p["min"] << ", max: " << p["max"] << ", step: " << p["step"] << std::endl;
    BruteforceParam<uint64_t> perfcounter(p["min"], p["max"], p["step"]);

    std::cout << "Finished parsing parameters from JSON." << std::endl;
    return BruteforceRange(pid, tids, gettickcount, filetime, perfcounter);
}

// If you don't want to check the entire keyspace
// this sets what portion of it to check.
// The keyspace indexes starts at 0
// You can use -1 for end to indicate the end of the keyspace
void BruteforceRange::limits(uint64_t start = 0, uint64_t end = static_cast<uint64_t>(-1))
{
    // TODO do it in a more optimal way
    start_when_ = start;

    // end is unsigned so technically this is 2^64 but the evaluation works just the same and -1 is the only negative number we need
    if (end != static_cast<uint64_t>(-1))
    {
        done_when_ = end;
        std::cout << "Will end the search early, once the state reaches " << done_when_;
    }
    std::cout << "Will stop once key is found or state reaches " << done_when_ << "\n";
}
