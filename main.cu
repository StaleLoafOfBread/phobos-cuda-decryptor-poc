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

// Set to 0 when the key is not yet found. Set to 1 once its found
__device__ volatile int found_key_gpu_volatile = 0;

// How many keys will each thread test, at least. Some threads may test 1 additional
__constant__ uint64_t baseKeysPerThread;

// How many keys are left over after each thread has tested baseKeysPerThread amount of keys
// This is how many threads will need to process an addition key
__constant__ uint64_t remainingKeys;

// Device Constant memory should be faster to access but
// 1) Is read only
// 2) Is only 64KB
__constant__ uint8_t ciphertext_gpu[16];
__constant__ uint8_t plaintext_cbc_gpu[16];
__constant__ const uint8_t key_high[] = {
    0x0d,
    0xdb,
    0x95,
    0x0c,
    0x33,
    0x68,
    0xc0,
    0xa0,
    0x06,
    0xe9,
    0x0c,
    0x24,
    0x44,
    0x88,
    0x1b,
    0x12,
};

template <typename T>
__device__ void deviceSwap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

// Variables that are read in from the json
__constant__ uint64_t perfcounter_min_d_constant;
__constant__ uint64_t perfcounter_keyspace_d_constant;

__constant__ uint16_t filetime_step_d_constant;
__constant__ uint64_t filetime_min_d_constant;
__constant__ uint64_t filetime_keyspace_d_constant;

__constant__ uint32_t pid_min_d_constant;
__constant__ uint8_t pid_and_tid_step_d_constant;
__constant__ uint32_t pid_keyspace_d_constant;

__constant__ uint32_t tid_min_d_constant;
__constant__ uint32_t tid_keyspace_d_constant;

__constant__ uint32_t pc_step_d_constant;
__constant__ uint32_t pc_mask_d_constant;
__constant__ uint32_t gtc_prefix_d_constant;

__constant__ uint64_t perfcounter_xor_keyspace_gpu_d_constant;

__host__ void perfcounter_xor_set_host(uint64_t *perfcounter_xor, uint64_t *perfcounter_xor_min, uint64_t *perfcounter_xor_max, uint64_t *min_pc, uint64_t min_gtc, uint64_t max_gtc)
{
    uint64_t pc_step = uint64_t{1} << variable_suffix_bits(min_gtc, max_gtc);
    uint64_t pc_mask = ~(pc_step - uint64_t{1});
    uint64_t gtc_prefix = (min_gtc & pc_mask);

    uint64_t max_pc = *min_pc + MAX_PERFCOUNTER_SECOND_CALL_TICKS_DIFF;

    uint64_t min_pc2 = *min_pc ^ gtc_prefix;
    uint64_t max_pc2 = max_pc ^ gtc_prefix;
    if (max_pc2 < min_pc2)
    {
        std::swap(min_pc2, max_pc2);
    }

    // Set the variables
    *perfcounter_xor_min = (min_pc2 & pc_mask);
    *perfcounter_xor_max = (max_pc2 & pc_mask) + pc_step;
    // *perfcounter_xor_keyspace = (*perfcounter_xor_max - *perfcounter_xor_min + 1);
    *perfcounter_xor = *perfcounter_xor_min;
}

__inline__ __device__ void perfcounter_xor_set_gpu(uint64_t *perfcounter_xor, uint64_t *perfcounter_xor_min, uint64_t *perfcounter_xor_max, uint64_t *min_pc, uint32_t *gtc_prefix, uint32_t *pc_mask, uint32_t *pc_step)
{
    const uint64_t max_pc = *min_pc + MAX_PERFCOUNTER_SECOND_CALL_TICKS_DIFF;

    uint64_t min_pc2 = *min_pc ^ *gtc_prefix;
    uint64_t max_pc2 = max_pc ^ *gtc_prefix;
    if (max_pc2 < min_pc2)
    {
        deviceSwap(min_pc2, max_pc2);
    }

    // Set the variables
    *perfcounter_xor_min = (min_pc2 & *pc_mask);
    *perfcounter_xor_max = (max_pc2 & *pc_mask) + *pc_step;
    *perfcounter_xor = *perfcounter_xor_min;
}

__global__ void process_packet(bool *found_key, uint8_t *device_final_key)
{
    // Store thread_id so we know what key to use
    const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // What key index to test. This
    // For example, thread 0 will always start at index 0 but thread 1 may start at 1 if each thread only handles 1 key.
    // Reason for being signed: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#loop-counters-signed-vs-unsigned
    int64_t key_index = thread_id * baseKeysPerThread + (thread_id < remainingKeys ? thread_id : remainingKeys);

    // Convert the constants to shared memory to avoid any coalescing issues
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
    // Info on why this shouldn't cause bank conflicts:
    // However, if multiple threads’ requested addresses map to the same memory bank, the accesses are serialized. The hardware splits a conflicting memory request into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of colliding memory requests. An exception is the case where all threads in a warp address the same shared memory address, resulting in a broadcast. Devices of compute capability 2.0 and higher have the additional ability to multicast shared memory accesses, meaning that multiple accesses to the same location by any number of threads within a warp are served simultaneously.
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    __shared__ uint64_t perfcounter_min;
    __shared__ uint64_t perfcounter_keyspace;

    __shared__ uint16_t filetime_step;
    __shared__ uint64_t filetime_min;
    __shared__ uint64_t filetime_keyspace;

    __shared__ uint32_t pid_min;
    __shared__ uint8_t pid_and_tid_step;
    __shared__ uint32_t pid_keyspace;

    __shared__ uint32_t tid_min;
    __shared__ uint32_t tid_keyspace;

    __shared__ uint32_t pc_step;
    __shared__ uint32_t pc_mask;
    __shared__ uint32_t gtc_prefix;

    __shared__ uint64_t perfcounter_xor_keyspace_gpu;

    // Set the variables from the first thread of the block
    if (threadIdx.x == 0)
    {
        perfcounter_min = perfcounter_min_d_constant;
        perfcounter_keyspace = perfcounter_keyspace_d_constant;

        filetime_step = filetime_step_d_constant;
        filetime_min = filetime_min_d_constant;
        filetime_keyspace = filetime_keyspace_d_constant;

        pid_min = pid_min_d_constant;
        pid_and_tid_step = pid_and_tid_step_d_constant;
        pid_keyspace = pid_keyspace_d_constant;

        tid_min = tid_min_d_constant;
        tid_keyspace = tid_keyspace_d_constant;

        pc_step = pc_step_d_constant;
        pc_mask = pc_mask_d_constant;
        gtc_prefix = gtc_prefix_d_constant;

        perfcounter_xor_keyspace_gpu = perfcounter_xor_keyspace_gpu_d_constant;
    }
    __syncthreads(); // We need to make sure all threads have the shared memory set before we continue

    // What key index to stop at
    // For example thead 0 will stop at index 1 if each thread only handles 1 key.
    // In that scenario thread 1 would stop at 2.
    // The thread will not test the key at the endIdx, instead the next thread will start there
    // Reason for being signed: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#loop-counters-signed-vs-unsigned
    const int64_t endIdx = key_index + baseKeysPerThread + (thread_id < remainingKeys ? 1 : 0);

    // Var to hold the data we will be running SHA and AES against
    uint32_t input_data[8];
    uint8_t decrypted_data_block_cbc[16];
    uint8_t key[32];
    aes256_context ctx;

    // Vars to control what key we are working on
    // key_index is what index the current key we are working on would be if all the keys were stored in an array
    // For example, 0 would be the first key with all inputs in their first value
    // Reason for being signed: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#loop-counters-signed-vs-unsigned
    int64_t key_index_adjusted = 0;

    // Since perfcounter_xor's range depends on the current value of perfcounter
    // we need to track it per thread
    // This section initializes vars for that purpose
    uint64_t perfcounter_xor_min;
    uint64_t perfcounter_xor_max;

    // Vars to control the current values of the various inputs that go into a key
    // Initialize them all to their lowest possible value as start low and work our way up
    uint64_t perfcounter_xor;
    uint64_t perfcounter = perfcounter_min;
    uint64_t filetime = filetime_min;
    uint32_t pid = pid_min;
    uint32_t tid = tid_min;
    perfcounter_xor_set_gpu(&perfcounter_xor, &perfcounter_xor_min, &perfcounter_xor_max, &perfcounter, &gtc_prefix, &pc_mask, &pc_step);

    // These star blocks are representing functions
    // We aren't using actual functions so that we can have no overhead
    // and directly pass around the data var
    // Perhaps using pointers would work just as well?

    // We keep checking unless the key was found, even by another thread
    // Within in the loop, if we exhaust our keyspace then we break
    __syncwarp();
    while (key_index < endIdx && !*found_key)
    {
        /*****************************************
         *                                       *
         *             START ROTATION            *
         *                                       *
         *****************************************/

        // Set the perfcounter then reset the perfcounter_xor because it may be different now
        perfcounter = (key_index % perfcounter_keyspace) + perfcounter_min;
        key_index_adjusted = key_index / perfcounter_keyspace;
        perfcounter_xor_set_gpu(&perfcounter_xor, &perfcounter_xor_min, &perfcounter_xor_max, &perfcounter, &gtc_prefix, &pc_mask, &pc_step);

        perfcounter_xor = perfcounter_xor_min + (key_index_adjusted % perfcounter_xor_keyspace_gpu);
        // If the perfcounter_xor should have maxed out
        // then increment the key_index_adjusted enough
        // so that it overflows the min value by how much it overflowed the true max
        // Ex: if the true range is from 0-100 but the fake range is 0-1000 then
        //     on key_index_adjusted being 130, we adjust it to 1030
        // We don't need this for the other inputs because their ranges are constant
        if (perfcounter_xor > perfcounter_xor_max)
        {
            key_index_adjusted += perfcounter_xor_keyspace_gpu - perfcounter_xor_max;
            perfcounter_xor = perfcounter_xor_min + (key_index_adjusted % perfcounter_xor_keyspace_gpu);
        }
        __syncwarp();
        key_index_adjusted = key_index_adjusted / perfcounter_xor_keyspace_gpu;

        // Get the filetime
        filetime = (key_index_adjusted % filetime_keyspace) * filetime_step + filetime_min;
        key_index_adjusted = key_index_adjusted / filetime_keyspace;

        // Get the pid
        pid = (key_index_adjusted % pid_keyspace) * pid_and_tid_step + pid_min;
        key_index_adjusted = key_index_adjusted / pid_keyspace;

        // Get the tid
        tid = (key_index_adjusted % tid_keyspace) * pid_and_tid_step + tid_min;

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

        /*****************************************
         *                                       *
         *              END ROTATION             *
         *                                       *
         *****************************************/

        /*****************************************
         *                                       *
         *               START SHA               *
         *                                       *
         *****************************************/

        input_data[0] = __byte_perm(input_data[0], 0, 0x123);
        input_data[1] = __byte_perm(input_data[1], 0, 0x123);
        input_data[2] = __byte_perm(input_data[2], 0, 0x123);
        input_data[3] = __byte_perm(input_data[3], 0, 0x123);
        input_data[4] = __byte_perm(input_data[4], 0, 0x123);
        input_data[5] = __byte_perm(input_data[5], 0, 0x123);
        input_data[6] = __byte_perm(input_data[6], 0, 0x123);
        input_data[7] = __byte_perm(input_data[7], 0, 0x123);

        // TODO: Check if the first round is always applied or not in original Phobos
        // This will always sha the first round and will always sha once more after it meets the criteria
        sha256_transform(input_data);
        while ((input_data[0] & 0xFF000000) != 0)
        {
            sha256_transform(input_data);
        }
        __syncwarp();
        sha256_transform(input_data);

        input_data[0] = __byte_perm(input_data[0], 0, 0x123);
        input_data[1] = __byte_perm(input_data[1], 0, 0x123);
        input_data[2] = __byte_perm(input_data[2], 0, 0x123);
        input_data[3] = __byte_perm(input_data[3], 0, 0x123);
        input_data[4] = __byte_perm(input_data[4], 0, 0x123);
        input_data[5] = __byte_perm(input_data[5], 0, 0x123);
        input_data[6] = __byte_perm(input_data[6], 0, 0x123);
        input_data[7] = __byte_perm(input_data[7], 0, 0x123);

        /*****************************************
         *                                       *
         *                END SHA                *
         *                                       *
         *****************************************/

        /*****************************************
         *                                       *
         *               START AES               *
         *                                       *
         *****************************************/

        // Upper half of the key is constant
        mycpy16((uint32_t *)key, (uint32_t *)input_data);
        mycpy16((uint32_t *)(key + 16), (uint32_t *)key_high);

        // Copy the encypted data into a block for us to then try to decrypt with our key
        mycpy16((uint32_t *)decrypted_data_block_cbc, (uint32_t *)ciphertext_gpu);

        // Run the decryption, saving the results into decrypted_data_block_cbc
        aes256_init(&ctx, key);
        aes256_decrypt_ecb(&ctx, decrypted_data_block_cbc);

        /*****************************************
         *                                       *
         *                END AES                *
         *                                       *
         *****************************************/

        /*****************************************
         *                                       *
         *            START VALIDATION           *
         *                                       *
         *****************************************/

        // Check if what we decrypted is identical to the plaintext
        // This is actually after its been xor'd with the initialization vector
        // which is denoted by the "cbc"
        // We do this to save the time xor'ing in this loop, squeezing out a bit more performance
        if (
            decrypted_data_block_cbc[0] == plaintext_cbc_gpu[0] &&
            decrypted_data_block_cbc[1] == plaintext_cbc_gpu[1] &&
            decrypted_data_block_cbc[2] == plaintext_cbc_gpu[2] &&
            decrypted_data_block_cbc[3] == plaintext_cbc_gpu[3] &&
            decrypted_data_block_cbc[4] == plaintext_cbc_gpu[4] &&
            decrypted_data_block_cbc[5] == plaintext_cbc_gpu[5] &&
            decrypted_data_block_cbc[6] == plaintext_cbc_gpu[6] &&
            decrypted_data_block_cbc[7] == plaintext_cbc_gpu[7] &&
            decrypted_data_block_cbc[8] == plaintext_cbc_gpu[8] &&
            decrypted_data_block_cbc[9] == plaintext_cbc_gpu[9] &&
            decrypted_data_block_cbc[10] == plaintext_cbc_gpu[10] &&
            decrypted_data_block_cbc[11] == plaintext_cbc_gpu[11] &&
            decrypted_data_block_cbc[12] == plaintext_cbc_gpu[12] &&
            decrypted_data_block_cbc[13] == plaintext_cbc_gpu[13] &&
            decrypted_data_block_cbc[14] == plaintext_cbc_gpu[14] &&
            decrypted_data_block_cbc[15] == plaintext_cbc_gpu[15])
        {
            // If we get this far that means we found it!
            // found_key_gpu_volatile = 1;
            *found_key = true;

            // Some debugging statements
            // printf("Thread %llu set found_key value.\n", thread_id);
            // printf("filetime = %llu\n", filetime);
            // printf("perfcounter = %llu\n", perfcounter);
            // printf("PID = %llu\n", pid);
            // printf("TID = %llu\n", tid);

            // Copy the key that worked into a var that will later be copied down to host
            mycpy16((uint32_t *)device_final_key, (uint32_t *)input_data);
            mycpy16((uint32_t *)(device_final_key + 16), (uint32_t *)key_high);

            // Debug statement showing the key
            //                 printf("AES Decryption Key [GPU]: 0x");
            // #pragma unroll 32
            //                 for (i = 0; i < 32; ++i)
            //                 {
            //                     printf("%02x", device_final_key[i]);
            //                 }
            //                 printf("\n");
        }
        __syncwarp();

        /*****************************************
         *                                       *
         *             END VALIDATION            *
         *                                       *
         *****************************************/

        // We fully tried this key so time to try the next one
        key_index++;
    }

    return;
}

class PhobosInstance
{
    Block16 plaintext_;
    Block16 iv_;
    Block16 ciphertext_;
    Block16 plaintex_cbc_;

public:
    PhobosInstance(Block16 plaintext, Block16 iv, Block16 ciphertext)
        : plaintext_(plaintext), iv_(iv), ciphertext_(ciphertext), plaintex_cbc_(plaintext)
    {
        // Bitwise XOR with the IV to start the Cipher Block Chaining for the entire plaintext
        for (int x = 0; x < 16; x++)
        {
            plaintex_cbc_[x] ^= iv[x];
        }

        std::cout << "Bytes of plaintex_cbc: 0x";
        for (int i = 0; i < 16; i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (0xff & static_cast<uint32_t>(plaintex_cbc_[i]));
        }
        std::cout << std::dec << std::endl;
    }

public:
    // Accessors for different block types:
    // The plaintext, the initial vector (iv), the ciphertext, and the plain text processed by CBC cipher
    const Block16 &plaintext() const { return plaintext_; }
    const Block16 &iv() const { return iv_; }
    const Block16 &ciphertext() const { return ciphertext_; }
    const Block16 &plaintext_cbc() const { return plaintex_cbc_; }

    // Static function to load plaintext and encrypted files
    static PhobosInstance load(const std::string &plain, const std::string &encrypted)
    {
        // Three block16 variables that hold plaintext, iv and ciphertext values
        Block16 plaintext, iv, ciphertext;

        // Open the plaintext file in binary mode
        std::ifstream plainf(plain, std::ios::binary);
        // Set to throw an exception if a stream error occurs
        plainf.exceptions(std::ifstream::badbit);
        // Read 16 bytes from the plaintext file into the Block16 'plaintext'
        plainf.read(reinterpret_cast<char *>(plaintext.data()), 16);

        // Open the encrypted file in binary mode
        std::ifstream cipherf(encrypted, std::ios::binary);
        // Set to throw an exception if a stream error occurs
        cipherf.exceptions(std::ifstream::badbit);
        // Read the first 16 bytes from the encrypted file into the Block16 'ciphertext'
        cipherf.read(reinterpret_cast<char *>(ciphertext.data()), 16);

        // Encrypted file format:
        //     data: N bytes
        //     footer (metadata): 178 bytes
        // Footer (metadata) format:
        //      padded w/ zeros: 0-20               (length: 20 bytes)
        //      iv: 20-36 bytes                     (length: 16 bytes)
        //      padded_size: 36-40 bytes            (length: 4 bytes)
        //      encrypted_key: 40-168 bytes         (length: 128 bytes)
        //      additional_data_size: 168-172 bytes (length: 4 bytes)
        //      attacker_id: 172-178 bytes          (length: 6 bytes)

        // Read in the IV by seek to the start of it then reading 16 bytes
        // 158 = 178 - 20
        // 158 = 6 + 4 + 128 + 4 + 16
        cipherf.seekg(-158, std::ios::end);
        cipherf.read(reinterpret_cast<char *>(iv.data()), 16);

        // Display all the bytes of the IV for debugging purposes
        std::cout << "Bytes of IV:           0x";
        for (int i = 0; i < 16; i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (0xff & static_cast<uint32_t>(iv[i]));
        }
        std::cout << std::dec << std::endl;

        // Display all the bytes of the plaintext for debugging purposes
        std::cout << "Bytes of plaintext:    0x";
        for (int i = 0; i < 16; i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (0xff & static_cast<uint32_t>(plaintext[i]));
        }
        std::cout << std::dec << std::endl;

        // Return a new PhobosInstance object created by the loaded plaintext, iv, and ciphertext
        return PhobosInstance(plaintext, iv, ciphertext);
    }
};

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

__host__ uint64_t get_perfcounter_xor_keyspace(uint64_t perfcounter_min, uint64_t perfcounter_max, std::vector<uint64_t> &keyspaceRecord, uint64_t min_gtc, uint64_t max_gtc)
{
    uint64_t perfcounter_xor = 0;
    uint64_t perfcounter_xor_min = 0;
    uint64_t perfcounter_xor_max = 0;

    // Vars to store the keyspace so we can find the largest one
    uint64_t perfcounter_xor_keyspace_max = 0;
    uint64_t perfcounter_xor_keyspace_cur = 0;

    for (uint64_t i = perfcounter_min; i <= perfcounter_max; i++)
    {
        // Calculate the xor range
        perfcounter_xor_set_host(&perfcounter_xor, &perfcounter_xor_min, &perfcounter_xor_max, &i, min_gtc, max_gtc);

        // Calculate the xor keyspace
        perfcounter_xor_keyspace_cur = (perfcounter_xor_max - perfcounter_xor_min + 1);

        // Record xor keyspace so we can later calculate the total keyspace
        keyspaceRecord.push_back(perfcounter_xor_keyspace_cur);

        // Record the highest keyspace we found so we know what use as its upperbound in GPU
        if (perfcounter_xor_keyspace_cur > perfcounter_xor_keyspace_max)
        {
            perfcounter_xor_keyspace_max = perfcounter_xor_keyspace_cur;
        }
    }

    return perfcounter_xor_keyspace_max;
}

std::string format_number(uint64_t value)
{
    std::stringstream ss;
    ss.imbue(std::locale("")); // Use the current locale to format numbers with thousands separators
    ss << value;
    return ss.str();
}

// Returns the total keyspace
uint64_t set_inputs_on_gpu()
{
    // Set the straight foward vars
    std::cout << "Setting input variables in GPU Memory" << std::endl;
    const uint64_t h_perfcounter_min = 19084705045;
    const uint64_t h_perfcounter_max = 19084705050;
    const uint64_t h_perfcounter_keyspace = (h_perfcounter_max - h_perfcounter_min + 1);
    cudaMemcpyToSymbol(perfcounter_min_d_constant, &h_perfcounter_min, sizeof(uint64_t));
    cudaMemcpyToSymbol(perfcounter_keyspace_d_constant, &h_perfcounter_keyspace, sizeof(uint64_t));
    assert(h_perfcounter_min <= h_perfcounter_max);

    const uint16_t h_filetime_step = 10000;
    const uint64_t h_filetime_min = 132489479687990000;
    const uint64_t h_filetime_max = 132489479687990000;
    const uint64_t h_filetime_keyspace = (h_filetime_max - h_filetime_min + h_filetime_step) / h_filetime_step;
    cudaMemcpyToSymbol(filetime_step_d_constant, &h_filetime_step, sizeof(uint16_t));
    cudaMemcpyToSymbol(filetime_min_d_constant, &h_filetime_min, sizeof(uint64_t));
    cudaMemcpyToSymbol(filetime_keyspace_d_constant, &h_filetime_keyspace, sizeof(uint64_t));
    assert(h_filetime_min <= h_filetime_max);
    assert(h_filetime_min % h_filetime_step == 0);
    assert(h_filetime_max % h_filetime_step == 0);

    const uint32_t h_pid_min = 3152;
    const uint32_t h_pid_max = 3152;
    const uint8_t h_pid_and_tid_step = 4;
    const uint32_t h_pid_keyspace = (h_pid_max - h_pid_min + h_pid_and_tid_step) / h_pid_and_tid_step;
    cudaMemcpyToSymbol(pid_min_d_constant, &h_pid_min, sizeof(uint32_t));
    cudaMemcpyToSymbol(pid_and_tid_step_d_constant, &h_pid_and_tid_step, sizeof(uint8_t));
    cudaMemcpyToSymbol(pid_keyspace_d_constant, &h_pid_keyspace, sizeof(uint32_t));
    assert(h_pid_min <= h_pid_max);
    assert(h_pid_min % h_pid_and_tid_step == 0);
    assert(h_pid_max % h_pid_and_tid_step == 0);

    const uint32_t h_tid_min = 1488;
    const uint32_t h_tid_max = 1492;
    const uint32_t h_tid_keyspace = (h_tid_max - h_tid_min + h_pid_and_tid_step) / h_pid_and_tid_step;
    cudaMemcpyToSymbol(tid_min_d_constant, &h_tid_min, sizeof(uint32_t));
    cudaMemcpyToSymbol(tid_keyspace_d_constant, &h_tid_keyspace, sizeof(uint32_t));
    assert(h_tid_min <= h_tid_max);
    assert(h_tid_min % h_pid_and_tid_step == 0);
    assert(h_tid_max % h_pid_and_tid_step == 0);

    // Initialize the vars for the second percounter call that's xor'd with the tick count
    std::cout << "Working on XOR Vars" << std::endl;
    const uint32_t h_min_gtc = 1910437;
    const uint32_t h_max_gtc = 1910437;
    const uint32_t h_pc_step = uint32_t{1} << variable_suffix_bits_optimized(h_min_gtc, h_max_gtc);
    const uint32_t h_pc_mask = ~(h_pc_step - uint32_t{1});
    const uint32_t h_gtc_prefix = (h_min_gtc & h_pc_mask);
    cudaMemcpyToSymbol(pc_step_d_constant, &h_pc_step, sizeof(uint32_t));
    cudaMemcpyToSymbol(pc_mask_d_constant, &h_pc_mask, sizeof(uint32_t));
    cudaMemcpyToSymbol(gtc_prefix_d_constant, &h_gtc_prefix, sizeof(uint32_t));
    assert(h_min_gtc <= h_max_gtc);

    std::cout << "Calculating perfcounter_xor keyspace" << std::endl;
    std::vector<uint64_t> recordedKeyspaces;
    uint64_t h_perfcounter_xor_keyspace = get_perfcounter_xor_keyspace(h_perfcounter_min, h_perfcounter_max, recordedKeyspaces, h_min_gtc, h_max_gtc);
    std::cout << "Maximum perfcounter_xor keyspace: " << format_number(h_perfcounter_xor_keyspace) << std::endl;
    cudaMemcpyToSymbol(perfcounter_xor_keyspace_gpu_d_constant, &h_perfcounter_xor_keyspace, sizeof(uint64_t));

    std::cout << "\n\nCalculating total keyspace" << std::endl;
    const uint64_t total_keyspace_no_xor = h_perfcounter_keyspace * h_filetime_keyspace * h_pid_keyspace * h_tid_keyspace;
    uint64_t total_keyspace = 0;
    for (const auto &value : recordedKeyspaces)
    {
        total_keyspace += value * total_keyspace_no_xor;
    }
    std::cout << "Total keyspace: " << format_number(total_keyspace) << "\n"
              << std::endl;
    return total_keyspace;
}

std::string formatDuration(int seconds)
{
    if (seconds < 0)
    {
        return "Invalid input";
    }

    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int remainingSeconds = seconds % 60;

    std::string result = std::to_string(hours) + "h " +
                         std::to_string(minutes) + "m " +
                         std::to_string(remainingSeconds) + "s";
    return result;
}

int brute(const PhobosInstance &phobos, BruteforceRange *range)
{
    // Record the start time to measure the overall execution time.
    auto gt1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nOkay, let's crack some keys!\n";

    // Copy plaintext_cbc data to GPU
    cudaMemcpyToSymbol(plaintext_cbc_gpu, phobos.plaintext_cbc().data(), sizeof(Block16));

    // Copy the ciphertext from the PhobosInstance from CPU to GPU.
    std::cout << "Copying the ciphertext from the PhobosInstance on CPU to GPU\n\n";
    cudaMemcpyToSymbol(ciphertext_gpu, phobos.ciphertext().data(), sizeof(Block16));

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

    // Copy inputs onto GPU memory and calculate the total keyspace
    uint64_t total_keyspace = set_inputs_on_gpu();

    // Debug lines during speed testing
    const uint64_t estimated_kps = 4028352;
    const uint64_t estimated_seconds = total_keyspace / estimated_kps;
    std::cout << "Assuming " << format_number(estimated_kps) << " keys per second this will take " << formatDuration(estimated_seconds) << std::endl;

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
    const int threadsPerBlock = 256; // getMaxThreadsPerBlock(0); // Set to the max threads per block for the first GPU seen. This may cause issues for clusters with mismatched GPUs

    // Sometimes forcing threads to calculate more than 1 key (aka using fewer blocks) can lead to better performance
    // Theres a theory that this should be a multiple of threadsPerBlock but that's just an unproven theory right now
    const int target_keys_per_thread = 256;

    // We set the total number of blocks such that there is at least one thread per key unless that exceeds the GPU's max, in which case we use the max
    const int numBlocks = ((int)(std::min((uint64_t)getMaxBlocks(0), (uint64_t)((total_keyspace + threadsPerBlock - 1) / threadsPerBlock)))) / target_keys_per_thread;

    const uint64_t totalThreads = (uint64_t)threadsPerBlock * (uint64_t)numBlocks;
    std::cout << "\n";
    std::cout << "Total Blocks:        " << numBlocks << "\n";
    std::cout << "Threads per block:   " << threadsPerBlock << "\n";
    std::cout << "Total Threads:       " << totalThreads << "\n";
    std::cout << "Max Keys Per Thread: " << ((total_keyspace + totalThreads - 1) / totalThreads) << "\n"; // Rounds up
    std::cout << "\n";

    // Ensure we don't waste any threads
    assert((threadsPerBlock % 32) == 0);

    // Set loop control vars for process_packet
    std::cout << "Copying loop control variables to GPU\n";
    uint64_t tmp_uint64_t = total_keyspace / totalThreads;
    cudaMemcpyToSymbol(baseKeysPerThread, &tmp_uint64_t, sizeof(uint64_t));
    if (print_cuda_errors())
    {
        return 99;
    }
    tmp_uint64_t = total_keyspace % totalThreads;
    cudaMemcpyToSymbol(remainingKeys, &tmp_uint64_t, sizeof(uint64_t));
    if (print_cuda_errors())
    {
        return 99;
    }

    // Record the start time for measuring how long it took to run
    t1 = std::chrono::high_resolution_clock::now();

    // Actually start brute forcing the key
    std::cout << "Starting the GPU Threads\n";
    process_packet<<<numBlocks, threadsPerBlock>>>(found_key_gpu, device_final_key);
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
    std::cout << "Keys Per Second: " << format_number(total_keyspace / seconds) << " kps" << std::endl;
    std::cout << "Keys Per Per Thread Per Second: " << format_number(total_keyspace / totalThreads / seconds) << std::endl;
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

void showCudaDeviceProp(int device_idx)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_idx);

    if (err != cudaSuccess)
    {
        std::cerr << "Error fetching device properties: " << cudaGetErrorString(err) << std::endl;
    }
    else
    {
        std::cout << "Device Name: " << deviceProp.name << std::endl;
        std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum shared memory available per block in bytes: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Maximum size for each dimension of a block: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Maximum size for each dimension of a grid: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    }

    return;
}

void showConfig()
{
    std::cout << "Usage Config:" << std::endl;
    std::cout << "BATCH_SIZE=" << BATCH_SIZE << std::endl;

    // TODO: Include more aspects of the program to get a more accurate estimate
    std::cout << "Estimated GPU Memory use (actual will be a few more bytes): " << ((BATCH_SIZE * BYTES_PER_KEY) / (1024.0 * 1024.0 * 1024.0)) << "GiB\n";

    const char *CUDA_VISIBLE_DEVICES = std::getenv("CUDA_VISIBLE_DEVICES");
    std::cout << "CUDA_VISIBLE_DEVICES=";
    if (CUDA_VISIBLE_DEVICES != nullptr)
    {
        std::cout << CUDA_VISIBLE_DEVICES << std::endl;
    }
    else
    {
        std::cout << "<NOT SET>" << std::endl;
    }

    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to retrieve number of CUDA devices: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Number of CUDA GPUs detected: " << nDevices << std::endl;

    showCudaDeviceProp(0);
}

int main(int argc, char *argv[])
{
    if (argc <= 2)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "./bruter show config" << std::endl;
        std::cout << "./bruter keyspace [config]" << std::endl;
        std::cout << "./bruter crack [config] [clear_file] [enc_file] [start] [end]" << std::endl;
        return 1;
    }

    if (std::string(argv[1]) == "keyspace")
    {
        BruteforceRange range = BruteforceRange::parse(argv[2]);
        std::cout << range.keyspace() << std::endl;
        return 0;
    }

    if (std::string(argv[1]) == "crack")
    {
        showConfig();
        std::cout << std::endl;
        BruteforceRange range = BruteforceRange::parse(argv[2]);
        char *endx;
        uint64_t start = std::strtoull(argv[5], &endx, 10);
        uint64_t end = std::strtoull(argv[6], &endx, 10);
        range.limits(start, end);
        PhobosInstance phobos = PhobosInstance::load(argv[3], argv[4]);
        return brute(phobos, &range);
    }

    if (std::string(argv[1]) == "show" && std::string(argv[2]) == "config")
    {
        showConfig();
        return 0;
    }

    std::cout << "No, I don't think I will" << std::endl;
    return 2;
}
