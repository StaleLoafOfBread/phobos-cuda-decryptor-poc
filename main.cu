#include <iostream>
#include <chrono>
#include <array>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <cassert>
#include "sha256.cuh"
#include "aes256.h"
#include "bruteforce_range.h"

#include <iomanip>
#include <ios>
#include <unordered_map>

// Device Constant memory should be faster to access but
// 1) Is read only
// 2) Is only 64KB
__constant__ uint64_t BATCH_SIZE_GPU = BATCH_SIZE;
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

__device__ void aes_decrypt(Packet packets[], PacketStatus statuses[])
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We spin up threads via blocks which have a specified number of threads each
    // Sometimes we end up with more threads than we need
    // because n is not evenly divisible by threads per block
    // This exits the thread if its an extra thread
    if (thread_id >= BATCH_SIZE_GPU)
    {
        return;
    }

    // Do not decrypt packets which have not yet finished having SHA ran against it
    if (statuses[thread_id] != PacketStatus::ReadyForAES)
    {
        return;
    }

    // Upper half of the key is constant
    uint8_t key[32];
    mycpy16((uint32_t *)key, (uint32_t *)packets[thread_id]);
    mycpy16((uint32_t *)(key + 16), (uint32_t *)key_high);

    uint8_t block[16];
    mycpy16((uint32_t *)block, (uint32_t *)ciphertext_gpu);

    aes256_context ctx;
    aes256_init(&ctx, key);
    aes256_decrypt_ecb(&ctx, block);

    mycpy16((uint32_t *)packets[thread_id], (uint32_t *)block);

    statuses[thread_id] = PacketStatus::ReadyForValidation;
}

__device__ void sha_rounds(Packet packets[], PacketStatus statuses[], bool *found_key)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We spin up threads via blocks which have a specified number of threads each
    // Sometimes we end up with more threads than we need
    // because n is not evenly divisible by threads per block
    // This exits the thread if its an extra thread
    if (thread_id >= BATCH_SIZE_GPU)
    {
        return;
    }

    // No need to run SHA against it if its already had it done and is waiting for AES
    if (statuses[thread_id] != PacketStatus::ReadyForSHA)
    {
        return;
    }

    uint32_t data[8];
    mycpy32(data, (uint32_t *)packets[thread_id]);

#pragma unroll 8
    for (int i = 0; i < 8; ++i)
    {
        data[i] = __byte_perm(data[i], 0, 0x123);
    }

    // TODO: Check if the first round is always applied or not
    bool is_done;
    for (int round = 0; round < MAX_SHA_ROUNDS_PER_GPU_CALL; round++)
    {
        is_done = (data[0] & 0xFF000000) == 0 && round != 0;
        sha256_transform(data);

        if (is_done)
        {
            break;
        }

        // Exit early if the key was already found by another thread
        if (*found_key)
        {
            return;
        }
    }

#pragma unroll 8
    for (int i = 0; i < 8; ++i)
        data[i] = __byte_perm(data[i], 0, 0x123);

    mycpy32((uint32_t *)packets[thread_id], data);

    // Mark the packet as ready for AES if we have finished running the SHA256_transform against it
    // Otherwise the status will remain as ReadyForSHA and we will pick up where we left off
    if (is_done)
    {
        statuses[thread_id] = PacketStatus::ReadyForAES;
        return;
    }
}

__global__ void process_packet(Packet packets[], PacketStatus statuses[], bool *found_key)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We spin up threads via blocks which have a specified number of threads each
    // Sometimes we end up with more threads than we need
    // because n is not evenly divisible by threads per block
    // This exits the thread if its an extra thread
    if (thread_id >= BATCH_SIZE_GPU)
    {
        return;
    }

    sha_rounds(packets, statuses, found_key);

    // Exit early if the key was already found by another thread while we were performing the sha_rounds
    if (*found_key)
    {
        return;
    }

    aes_decrypt(packets, statuses);

    // Exit early if the key was already found by another thread while we were performing the aws_decryption
    if (*found_key)
    {
        return;
    }
    // Check if we have found the key
    // This will set the found_key to true if appropiate
    if (statuses[thread_id] == PacketStatus::ReadyForValidation)
    {
        // Actually perform the validation
        if (
            packets[thread_id][0] != plaintext_cbc_gpu[0] ||
            packets[thread_id][1] != plaintext_cbc_gpu[1] ||
            packets[thread_id][2] != plaintext_cbc_gpu[2] ||
            packets[thread_id][3] != plaintext_cbc_gpu[3] ||
            packets[thread_id][4] != plaintext_cbc_gpu[4] ||
            packets[thread_id][5] != plaintext_cbc_gpu[5] ||
            packets[thread_id][6] != plaintext_cbc_gpu[6] ||
            packets[thread_id][7] != plaintext_cbc_gpu[7] ||
            packets[thread_id][8] != plaintext_cbc_gpu[8] ||
            packets[thread_id][9] != plaintext_cbc_gpu[9] ||
            packets[thread_id][10] != plaintext_cbc_gpu[10] ||
            packets[thread_id][11] != plaintext_cbc_gpu[11] ||
            packets[thread_id][12] != plaintext_cbc_gpu[12] ||
            packets[thread_id][13] != plaintext_cbc_gpu[13] ||
            packets[thread_id][14] != plaintext_cbc_gpu[14] ||
            packets[thread_id][15] != plaintext_cbc_gpu[15])
        {
            statuses[thread_id] = PacketStatus::ReadyForRotation;
        }
        else
        {
            // If we get this far that means we found it!
            *found_key = true;
            printf("Thread %d set found_key value: %d\n", thread_id, *found_key);
            printf("plaintext_cbc: ");
            for (int i = 0; i < 32; ++i)
            {
                printf("%02x", packets[thread_id][i]);
            }
            printf("\n");
            statuses[thread_id] = PacketStatus::KeySpaceExhaustedOrKeyFound;
            return;
        }
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

// Rotate finished keys. Returns true if the full range is scanned.
bool rotate_keys(BruteforceRange *range, Packet packets[], PacketStatus statuses[], uint32_t size)
{
    bool any_tasks_in_progress = false;

    for (int i = 0; i < size; i++)
    {
        if (statuses[i] != PacketStatus::ReadyForRotation)
        {
            // There is at least one packet still being processed
            if (statuses[i] != PacketStatus::KeySpaceExhaustedOrKeyFound)
            {
                any_tasks_in_progress = true;
            }

            // Process next packet
            continue;
        }

        // Perform the rotation and enter statement if it couldn't rotate
        if (!range->next(packets[i], &statuses[i], i))
        {
            // There are no more possible combinations to try but there could still be some combinations being processed
            statuses[i] = PacketStatus::KeySpaceExhaustedOrKeyFound;

            // Process next packet
            // If we return here we won't evaluate the packets that have yet to finish being processed
            continue;
        }
        else
        {
            // We could rotate this one so that means there is at least one packet which is still being processed
            any_tasks_in_progress = true;
        }
    }

    // Return whether or not the full range has been processed
    return !any_tasks_in_progress;
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

// Prints the count of each status
void countStatuses(const PacketStatus statuses[], const size_t n)
{
    std::unordered_map<PacketStatus, size_t> status_counts;

    // Initialize all possible statuses to 0.
    status_counts[PacketStatus::ReadyForSHA] = 0;
    status_counts[PacketStatus::ReadyForAES] = 0;
    status_counts[PacketStatus::ReadyForValidation] = 0;
    status_counts[PacketStatus::ReadyForRotation] = 0;
    status_counts[PacketStatus::KeySpaceExhaustedOrKeyFound] = 0;
    // ... initialize counts for any additional statuses to 0 ...

    // Count occurrences of each status.
    for (size_t i = 0; i < n; ++i)
    {
        status_counts[statuses[i]]++;
    }

    // Output the counts.
    std::cout << "Status counts:\n";
    std::cout << "ReadyForSHA:                 " << status_counts[PacketStatus::ReadyForSHA] << "\n";
    std::cout << "ReadyForAES:                 " << status_counts[PacketStatus::ReadyForAES] << "\n";
    std::cout << "ReadyForValidation:          " << status_counts[PacketStatus::ReadyForValidation] << "\n";
    std::cout << "ReadyForRotation:            " << status_counts[PacketStatus::ReadyForRotation] << "\n";
    std::cout << "KeySpaceExhaustedOrKeyFound: " << status_counts[PacketStatus::KeySpaceExhaustedOrKeyFound] << "\n";
    // ... output counts for any additional statuses ...
}

int brute(const PhobosInstance &phobos, BruteforceRange *range)
{
    // Record the start time to measure the overall execution time.
    auto gt1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nOkay, let's crack some keys!\n";
    std::cout << "Total keyspace: " << range->keyspace() << "\n\n";

    // Define data structures for packets on both CPU and GPU.
    Packets packets_gpu, packets_cpu;

    // Allocate memory for packet data on CPU and GPU.
    std::cout << "Allocating memory for packet data on CPU and GPU\n";
    cudaMallocHost(&packets_cpu.data, BATCH_SIZE * sizeof(Packet));
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaMalloc(&packets_gpu.data, BATCH_SIZE * sizeof(Packet));
    if (print_cuda_errors())
    {
        return 99;
    }

    // Allocate memory for packet statuses on CPU and GPU.
    std::cout << "Allocating memory for packet statuses on CPU and GPU\n";
    cudaMallocHost(&packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus));
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaMalloc(&packets_gpu.statuses, BATCH_SIZE * sizeof(PacketStatus));
    if (print_cuda_errors())
    {
        return 99;
    }

    // Copy plaintext_cbc data to GPU
    cudaMemcpyToSymbol(plaintext_cbc_gpu, phobos.plaintext_cbc().data(), sizeof(Block16));

    // Initialise all packets
    std::cout << "Initializing all packet statuses\n";
    for (int x = 0; x < BATCH_SIZE; x++)
    {
        packets_cpu.statuses[x] = PacketStatus::ReadyForRotation;
    }

    // Copy packet data and statuses from CPU to GPU.
    std::cout << "Copying packet statuses from CPU to GPU\n";
    cudaMemcpy(packets_gpu.statuses, packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyHostToDevice);
    if (print_cuda_errors())
    {
        return 99;
    }

    // Copy the ciphertext from the PhobosInstance from CPU to GPU.
    std::cout << "Copying the ciphertext from the PhobosInstance on CPU to GPU\n\n";
    cudaMemcpyToSymbol(ciphertext_gpu, phobos.ciphertext().data(), sizeof(Block16));

    // Delcare outside the loop to prevent reinitialization
    // Compiler might be doing this under the hood but just in case...
    float percent = 0.0f;
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

    // Define the blocks/threads for the CUDA/Device/Kernal/GPU functions
    // AKA the Launch Configuration
    const float adjustment_for_resource_error = 1.75;                                            // If we don't adjust we get an error "too many resources requested for launch". Really not sure why but reducing threads per block fixes it so thats the bandaid for now as we want the threads to be dynamic based on BATCH_SIZE to ensure there is a thread per packet
    const int threadsPerBlock = (float)getMaxThreadsPerBlock(0) / adjustment_for_resource_error; // Set to the max threads per block for the first GPU seen. This may cause issues for clusters with mismatched GPUs
    const int numBlocks = ((BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock);                // This rounds up to ensure all elements are processed.
    std::cout << "Total Blocks:      " << numBlocks << "\n";
    std::cout << "Threads per block: " << threadsPerBlock << "\n";
    std::cout << "Total Threads:     " << (threadsPerBlock * numBlocks) << "\n";
    assert((threadsPerBlock * numBlocks) >= BATCH_SIZE); // Ensure that we always have enough threads for each packet we are working on

    // Calculate the percentage of progress and display the current state.
    percent = range->progress() * 100.0;
    std::cout << "\nState: " << range->current() << "/" << range->done_when() << " (" << percent << "%)\n";
    std::cout << "Total Keys Tried: " << range->total_keys_tried() << "\n";
    std::cout << "You may see the program get stuck at a state/percentage. This is expected and due to it still processing some packets. The state increases as a new attempt is started, not when its finished.\n";

    // Initalize all the packets with data before entering the loop
    std::cout << "Initializing all the packets\n";
    rotate_keys(range, packets_cpu.data, packets_cpu.statuses, BATCH_SIZE);
    cudaMemcpyAsync(packets_gpu.data, packets_cpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyHostToDevice);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaMemcpyAsync(packets_gpu.statuses, packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyHostToDevice);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaDeviceSynchronize();

    // Start the brute-force attack
    while (true)
    {

        // Record the start time for measuring the duration of each batch.
        t1 = std::chrono::high_resolution_clock::now();

        // Calculate the percentage of progress and display the current state.
        percent = range->progress() * 100.0;
        std::cout << "\nState: " << range->current() << "/" << range->done_when() << " (" << percent << "%)\n";
        std::cout << "Total Keys Tried/Currently Being Processed: " << range->total_keys_tried() << "\n";

        process_packet<<<numBlocks, threadsPerBlock>>>(packets_gpu.data, packets_gpu.statuses, found_key_gpu);
        if (print_cuda_errors())
        {
            return 99;
        }

        std::cout << "Copying GPU results to CPU for found flag\n";
        cudaMemcpy(&found_key, found_key_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
        if (print_cuda_errors())
        {
            return 99;
        }
        if (found_key)
        {
            std::cout << "Found needle in GPU\n";
            found_key = true;
            break;
        }

        // Copy the results from GPU to CPU
        // We need to copy the full data from the GPU as some SHA hashing may be incomplete and later we overwrite it when we rotate the keys so we neeed to the original value so that we don't change the value when we overwrite it
        std::cout << "Copying GPU packets and statuses to CPU so we can rotate them\n";
        cudaMemcpyAsync(packets_cpu.data, packets_gpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyDeviceToHost);
        if (print_cuda_errors())
        {
            return 99;
        }
        cudaMemcpyAsync(packets_cpu.statuses, packets_gpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyDeviceToHost);
        if (print_cuda_errors())
        {
            return 99;
        }
        cudaDeviceSynchronize();

        // Rotate keys and check if the full range is scanned.
        std::cout << "Rotating keys\n";
        if (rotate_keys(range, packets_cpu.data, packets_cpu.statuses, BATCH_SIZE))
        {
            std::cout << "Keyspace exhausted and nothing found\n";
            break;
        }

        // Copy the next batch of tasks from CPU to GPU asynchronously.
        std::cout << "Copying next batch to GPU!\n";
        cudaMemcpyAsync(packets_gpu.data, packets_cpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyHostToDevice);
        if (print_cuda_errors())
        {
            return 99;
        }
        cudaMemcpyAsync(packets_gpu.statuses, packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyHostToDevice);
        if (print_cuda_errors())
        {
            return 99;
        }
        cudaDeviceSynchronize();

        // Record the end time of the batch and calculate its duration.
        t2 = std::chrono::high_resolution_clock::now();
        duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "Batch total time: " << ((float)duration2 / 1000000) << "s" << std::endl
                  << std::endl;
    }

    // Record the end time of the brute-force attack and calculate total duration.
    auto gt2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(gt2 - gt1).count();
    std::cout << "\nTotal time: " << ((float)duration / 1000000) << std::endl;
    countStatuses(packets_cpu.statuses, BATCH_SIZE);

    // Calculate the percentage of progress and display the current state.
    percent = range->progress() * 100.0;
    std::cout << "\nState: " << range->current() << "/" << range->done_when() << " (" << percent << "%)\n";
    std::cout << "Total Keys Tried: " << range->total_keys_tried() << "\n";

    // Free allocated memory on CPU and GPU.
    cudaFreeHost(packets_cpu.data);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaFree(packets_gpu.data);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaFreeHost(packets_cpu.statuses);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaFree(packets_gpu.statuses);
    if (print_cuda_errors())
    {
        return 99;
    }
    cudaFree(found_key_gpu);
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
