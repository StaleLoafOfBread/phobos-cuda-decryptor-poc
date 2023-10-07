#pragma once

#include <iostream>
#include <chrono>
#include <array>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <cassert>

enum class PacketStatus : uint8_t
{
    ReadyForSHA,
    ReadyForAES,
    ReadyForValidation,
    ReadyForRotation,
    KeySpaceExhausted
};

typedef unsigned char Packet[32];

struct Packets
{
    PacketStatus *statuses;
    Packet *data;
};

using Block16 = std::array<uint8_t, 16>;

const int SHA_ROUNDS = 64;

const int BATCH_SIZE = 16 * 1024 * 1024; // How many keys to try concurrently
// const unsigned int PACKETS_SIZE = BATCH_SIZE * sizeof(Packet);

const uint64_t MAX_PERFCOUNTER_SECOND_CALL_TICKS_DIFF = 1000;

template <typename T>
class BruteforceParam
{
    T min_;
    T max_;
    T current_;
    T step_;

public:
    BruteforceParam(T min, T max, T step = 1) : min_(min), max_(max), current_(min), step_(step)
    {
        assert(step != 0);
        assert(min <= max);
        assert((max - min) % step == 0);
    }

    T get() const { return current_; }

    T min() const { return min_; }

    T max() const { return max_; }

    T keyspace() const { return (max_ - min_ + step_) / step_; }

    /**
     * Increases current value. Returns false on overflow.
     * Resets to min on overflow
     *
     * @return true when able to move to next iteration. False on overflow.
     */
    bool next()
    {
        if (current_ < max_)
        {
            current_ += step_;
            return true;
        }
        current_ = min_;
        return false;
    }
};

class BruteforceRange
{
    // For work progress calculation
    uint32_t total_keys_tried_;   // How many keys have been tried or more accurately, how many have been returned for another function to try
    uint32_t cur_keyspace_index_; // What key of the current keyspace are we on. This is not the key itself. Starts at 0.
    uint32_t keyspace_;           // How many total keys there are to check not counting percounter_xor because I couldn't figure out that math and its better use of my time to optimize than make the progress bar more precise

    uint32_t pid_;

    std::vector<uint32_t> tids_;

    uint64_t start_when_; // start bruting at this chunk. To start at the begining set to 0
    uint64_t done_when_;  // stop bruting at this chunk. Remember that this is array based so we start at 0 and end at the total count minus 1 (or whatever the user requested)
    bool no_more_keys_;   // Will be set to true once we exhausted all the keys

    BruteforceParam<uint32_t> gettickcount_;
    BruteforceParam<uint64_t> filetime_;
    BruteforceParam<uint64_t> perfcounter_;
    BruteforceParam<uint64_t> perfcounter_xor_;

    void perfcounter_xor_set();

    // Moves the internal state one step forward. Returns false if the range is done.
    bool forward();

public:
    BruteforceRange(uint32_t pid, std::vector<uint32_t> tids, BruteforceParam<uint32_t> gettickcount,
                    BruteforceParam<uint64_t> filetime, BruteforceParam<uint64_t> perfcounter);

    bool next(Packet target, PacketStatus *status, int thread_id);

    uint64_t current() const;

    uint64_t keyspace() const;

    uint64_t done_when() const;

    uint64_t total_keys_tried() const;

    float progress() const;

    static BruteforceRange parse(std::string path);

    void limits(uint64_t start, uint64_t end);
};
