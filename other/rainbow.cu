#include <iostream>
#include <chrono>
#include <array>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <cassert>
#include "sha256.cuh"
#include "aes256.h"
#include "bruteforce_range.cuh"

#include <iomanip>
#include <ios>
#include <unordered_map>

#include <string>
#include <locale>
#include <sstream>

#include <inttypes.h>

__global__ void rainbow()
{
    const uint64_t perfcounter_xor = 19084705000;
    const uint64_t perfcounter = 19084705000;
    const uint64_t filetime = 132489479687990000;
    const uint32_t pid = 3152;
    const uint32_t tid = 3152;

    uint32_t input_data[8];

    // Actually set the keys into one var
    //                                          // These comments represent what the data was in the original Phobos
    input_data[0] = perfcounter_xor;                  // Second call to QueryPerformanceCounter() xor'd against GetTickCount(). The reason its the second call despite being the first key is in the original Phobos, it would call GetTickCount() first then later XOR it with a new call to QueryPerformanceCounter()
    input_data[1] = (perfcounter >> 32) & 0xFFFFFFFF; // First call to QueryPerformanceCounter()
    input_data[2] = perfcounter & 0xFFFFFFFF;         // First call to QueryPerformanceCounter()
    input_data[3] = pid;                              // GetCurrentProcessId()
    input_data[4] = tid;                              // GetCurrentThreadId()
    input_data[5] = (filetime >> 32) & 0xFFFFFFFF;    // GetLocalTime() + SystemTimeToFileTime()
    input_data[6] = filetime & 0xFFFFFFFF;            // GetLocalTime() + SystemTimeToFileTime()
    input_data[7] = (perfcounter >> 32) & 0xFFFFFFFF; // Second call to QueryPerformanceCounter(). I think the idea behind this is that sinces its the high bits, there is a "high probability" that the first call and second call to QueryPerformanceCounter() will have the same high bits // with high probability

    sha256_transform(input_data);
    return;
}

// Prints the last cuda error to screen
// Returns true if there were any errors to print
// otherwise returns false
bool print_cuda_errors()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return true;
    }
    return false;
}

int getMaxThreadsPerBlock(int device_id)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);

    if (err != cudaSuccess)
    {
        std::cerr << "Error fetching device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return deviceProp.maxThreadsPerBlock;
}

int getMaxBlocks(int device_id)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);

    if (err != cudaSuccess)
    {
        std::cerr << "Error fetching device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return deviceProp.maxGridSize[0];
}

std::string format_number(uint64_t value)
{
    std::stringstream ss;
    ss.imbue(std::locale("")); // Use the current locale to format numbers with thousands separators
    ss << value;
    return ss.str();
}

std::string formatDuration(uint64_t seconds)
{
    uint64_t hours = seconds / 3600;
    uint64_t minutes = (seconds % 3600) / 60;
    uint64_t remainingSeconds = seconds % 60;

    std::string result = std::to_string(hours) + "h " +
                         std::to_string(minutes) + "m " +
                         std::to_string(remainingSeconds) + "s";
    return result;
}

int main(int argc, char *argv[])
{
    // Record the start time to measure the overall execution time.
    auto gt1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nOkay, let's crack some keys!\n";

    // Delcare outside the loop to prevent reinitialization
    // Compiler might be doing this under the hood but just in case...
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    bool found_key = false;

    // Allocate and initialize variable for tracking if we found the key on the gpu
    bool *found_key_gpu;
    cudaMalloc(&found_key_gpu, sizeof(bool));
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaMemset(found_key_gpu, false, sizeof(bool));
    if (print_cuda_errors())
    {
        return 99;
    }

    uint8_t *host_final_key = (uint8_t *)malloc(32 * sizeof(uint8_t));
    uint8_t *device_final_key;
    cudaMalloc((void **)&device_final_key, 32 * sizeof(uint8_t));
    if (print_cuda_errors())
    {
        return 99;
    }

    // Define the blocks/threads for the CUDA/Device/Kernal/GPU functions
    // AKA the Launch Configuration
    // const float adjustment_for_resource_error = 1.75; // If we don't adjust we get an error "too many resources requested for launch". Really not sure why but reducing threads per block fixes it so thats the bandaid for now as we want the threads to be dynamic based on BATCH_SIZE to ensure there is a thread per packet
    // const int threadsPerBlock = 640;                                              // (float)getMaxThreadsPerBlock(0) / adjustment_for_resource_error; // Set to the max threads per block for the first GPU seen. This may cause issues for clusters with mismatched GPUs
    // const int numBlocks = ((BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock); // This rounds up to ensure all elements are processed.

    // Threads per block should be a multiple of warp size to avoid wasting computation on under-populated warps and to facilitate coalescing.
    // A minimum of 64 threads per block should be used, and only if there are multiple concurrent blocks per multiprocessor.
    // Between 128 and 256 threads per block is a good initial range for experimentation with different block sizes.
    // Use several smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics
    const int threadsPerBlock = getMaxThreadsPerBlock(0); // Set to the max threads per block for the first GPU seen. This may cause issues for clusters with mismatched GPUs

    // Sometimes forcing threads to calculate more than 1 key (aka using fewer blocks) can lead to better performance
    // Theres a theory that this should be a multiple of threadsPerBlock but that's just an unproven theory right now
    const int target_keys_per_thread = 1;

    // We set the total number of blocks such that there is at least one thread per key unless that exceeds the GPU's max, in which case we use the max
    int numBlocks = (uint64_t)getMaxBlocks(0) / 2;

    const uint64_t totalThreads = (uint64_t)threadsPerBlock * (uint64_t)numBlocks;
    std::cout << "\n";
    std::cout << "Total Blocks:        " << numBlocks << "\n";
    std::cout << "Threads per block:   " << threadsPerBlock << "\n";
    std::cout << "Total Threads:       " << totalThreads << "\n";
    std::cout << "\n";

    // Ensure we don't waste any threads
    assert((threadsPerBlock % 32) == 0);

    // Ensure we don't accidently try 0 threads
    assert(totalThreads > 0);

    // Record the start time for measuring how long it took to run
    t1 = std::chrono::high_resolution_clock::now();

    // Actually start brute forcing the key
    std::cout << "Starting the GPU Threads\n";
    rainbow<<<numBlocks, threadsPerBlock>>>();
    if (print_cuda_errors())
    {
        return 99;
    }
    std::cout << "Waiting for the GPU Threads\n";
    cudaDeviceSynchronize();
    if (print_cuda_errors())
    {
        return 99;
    }

    // Copy down the flag that is set when we have found the key
    cudaMemcpy(&found_key, found_key_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
    if (print_cuda_errors())
    {
        return 99;
    }

    // Check if we found the key and if so, present it to the user
    if (found_key)
    {
        std::cout << "Key found! Copying key from GPU...\n";
        cudaMemcpy(host_final_key, device_final_key, 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (print_cuda_errors())
        {
            return 99;
        }

        std::cout << "AES Decryption Key [CPU]: 0x" << std::hex;
        for (int i = 0; i < 32; ++i)
        {
            std::cout << std::setw(2) << std::setfill('0')
                      << static_cast<uint32_t>((host_final_key)[i]);
        }
        std::cout << std::dec << std::endl;
    }
    else
    {
        std::cout << "The key was not found.\n";
    }

    // Record the end time of the brute-force attack and calculate total duration.
    auto gt2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(gt2 - gt1).count();
    double seconds = ((double)duration / 1000000);
    std::cout << "\nTotal time: " << formatDuration(seconds) << std::endl;
    if (found_key)
    {
        std::cout << "Warning: Since the key was found, not all keys in the keyspace were actually tested, thus making the speed measurements inaccurate representations of how fast this GPU can try keys." << std::endl;
    }
    std::cout << std::endl
              << std::endl;

    // Free memory on GPU
    cudaDeviceReset();
    if (print_cuda_errors())
    {
        return 99;
    }

    // Return based on if the key was found
    if (found_key)
    {
        return 0;
    }
    else
    {
        return 404;
    }
}
