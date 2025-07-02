#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <windows.h>

using namespace std;

struct result
{
    uint32_t foo;
    uint32_t diff;
};

result key_sched()
{
    uint32_t kk0, kk1, kk2, kk3, kk4, kk5, kk6, kk7 = 0;

    struct _SYSTEMTIME SystemTime;
    struct _FILETIME FileTime;
    LARGE_INTEGER PerformanceCount1;
    LARGE_INTEGER PerformanceCount2;
    QueryPerformanceCounter(&PerformanceCount1);
    kk0 ^= GetTickCount();
    kk1 ^= PerformanceCount1.HighPart;
    kk2 ^= PerformanceCount1.LowPart;
    kk3 ^= GetCurrentProcessId();
    kk4 ^= GetCurrentThreadId();
    GetLocalTime(&SystemTime);
    SystemTimeToFileTime(&SystemTime, &FileTime);
    kk5 ^= FileTime.dwHighDateTime;
    kk6 ^= FileTime.dwLowDateTime;
    QueryPerformanceCounter(&PerformanceCount2);
    // kk7 ^= PerformanceCount.HighPart;
    // kk0 ^= PerformanceCount.LowPart;

    auto foo = kk0 + kk1 + kk2 + kk3 + kk4 + kk5 + kk6;
    auto diff = PerformanceCount2.QuadPart - PerformanceCount1.QuadPart;
    // cout << PerformanceCount2.QuadPart << endl;
    // cout << PerformanceCount1.QuadPart << endl;
    // cout << diff << endl;
    // cout << endl;

    result res;
    res.diff = diff;
    res.foo = foo;

    return res;
}

int main(int argc, char *argv[])
{
    int test_count = 100000000;
    cout << "Beginning test for second perfcounter call delta\n";

    vector<uint64_t> results;

    // Call key_sched() 100 times and store the results in the vector
    cout << "Running " << test_count << " tests" << endl;
    for (int i = 0; i < test_count; ++i)
    {
        result result = key_sched();
        results.push_back(result.diff);
        // cout << result << endl;
    }

    // Display the min, max, average, and mode
    cout << "Tests complete" << endl;
    if (!results.empty())
    {
        // Find min, max, and average
        auto min_val = *min_element(results.begin(), results.end());
        auto max_val = *max_element(results.begin(), results.end());
        auto avg_val = accumulate(results.begin(), results.end(), uint64_t{}) / results.size();

        cout << "Min: " << min_val << endl;
        cout << "Max: " << max_val << endl;
        cout << "Average: " << avg_val << endl;

        // Find mode
        map<uint64_t, int> frequency_map;
        for (const auto &result : results)
        {
            frequency_map[result]++;
        }

        auto mode_pair = *max_element(frequency_map.begin(), frequency_map.end(),
                                      [](const pair<uint64_t, int> &p1, const pair<uint64_t, int> &p2)
                                      {
                                          return p1.second < p2.second;
                                      });

        cout << "Mode: " << mode_pair.first << " (occurs " << mode_pair.second << " times)" << endl;
    }

    return 0;
}
