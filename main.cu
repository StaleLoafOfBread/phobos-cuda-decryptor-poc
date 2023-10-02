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
__device__ const uint8_t key_high[] = {
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

__global__ void aes_decrypt(unsigned int n, Packet packets[], PacketStatus statuses[], uint8_t *ciphertext)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= n)
    {
        return;
    }

    // do not decrypt unfinished packets
    if (statuses[thread_id] != PacketStatus::Done)
    {
        return;
    }

    // Upper half of the key is constant
    uint8_t key[32];
    mycpy16((uint32_t *)key, (uint32_t *)packets[thread_id]);
    mycpy16((uint32_t *)(key + 16), (uint32_t *)key_high);

    uint8_t block[16];
    mycpy16((uint32_t *)block, (uint32_t *)ciphertext);

    aes256_context ctx;
    aes256_init(&ctx, key);
    aes256_decrypt_ecb(&ctx, block);

    mycpy16((uint32_t *)packets[thread_id], (uint32_t *)block);
}

__global__ void sha_rounds(unsigned int n, Packet packets[], PacketStatus statuses[])
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (statuses[thread_id] == PacketStatus::Done)
    {
        return;
    }

    if (thread_id >= n)
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
    for (int round = 0; round < SHA_ROUNDS; round++)
    {
        bool is_done = (data[0] & 0xFF000000) == 0 && round != 0;
        sha256_transform(data);

        if (is_done)
        {
            statuses[thread_id] = PacketStatus::Done;
            break;
        }
    }

#pragma unroll 8
    for (int i = 0; i < 8; ++i)
        data[i] = __byte_perm(data[i], 0, 0x123);

    mycpy32((uint32_t *)packets[thread_id], data);
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

// Checks if we found the needle. Returns true if the work is done.
bool find_needle(const PhobosInstance &phobos, Packet packets[], PacketStatus statuses[], uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        if (statuses[i] != PacketStatus::Done)
        {
            continue;
        }

        // check if decrypted value matches the iv-xored plaintext
        if (memcmp(packets[i], phobos.plaintext_cbc().data(), 16) == 0)
        {
            std::cout << "Found... something?\n";
            for (int q = 0; q < 32; q++)
            {
                printf("%02x", packets[i][q]);
            }
            std::cout << ("\n");

            // We found the key so return true
            return true;
        }
    }
    return false;
}

// Rotate finished keys. Returns true if the full range is scanned.
bool rotate_keys(BruteforceRange *range, Packet packets[], PacketStatus statuses[], uint32_t size)
{
    bool any_tasks_in_progress = false;
    for (int i = 0; i < size; i++)
    {
        if (statuses[i] != PacketStatus::Done)
        {
            any_tasks_in_progress = true;
            continue;
        }
        if (!range->next(packets[i], &statuses[i]))
        {
            std::cout << "No more things to try!\n";
            return !any_tasks_in_progress;
        }
        any_tasks_in_progress = true;
    }
    return !any_tasks_in_progress;
}

void print_cuda_errors()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

int brute(const PhobosInstance &phobos, BruteforceRange *range)
{

    // Record the start time to measure the overall execution time.
    auto gt1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nOkay, let's crack some keys!\n";
    std::cout << "Total keyspace: " << range->keyspace() << "\n\n";

    // Define data structures for packets and ciphertext on both CPU and GPU.
    Packets packets_gpu, packets_cpu;
    uint8_t *ciphertext_gpu;

    // Allocate memory for packet data on CPU and GPU.
    std::cout << "Allocating memory for packet data on CPU and GPU\n";
    cudaMallocHost(&packets_cpu.data, BATCH_SIZE * sizeof(Packet));
    cudaMalloc(&packets_gpu.data, BATCH_SIZE * sizeof(Packet));

    // Allocate memory for packet statuses on CPU and GPU.
    std::cout << "Allocating memory for packet statuses on CPU and GPU\n";
    cudaMallocHost(&packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus));
    cudaMalloc(&packets_gpu.statuses, BATCH_SIZE * sizeof(PacketStatus));

    // Allocate memory for ciphertext on GPU.
    std::cout << "Allocating memory for ciphertext on GPU\n";
    cudaMalloc(&ciphertext_gpu, 16);

    // Initialise all packets to the 'Done' state.
    std::cout << "Initializing all packets\n";
    for (int x = 0; x < BATCH_SIZE; x++)
    {
        packets_cpu.statuses[x] = PacketStatus::Done;
    }

    // Copy packet data and statuses from CPU to GPU.
    std::cout << "Copying packet data and statuses from CPU to GPU\n";
    cudaMemcpy(packets_gpu.data, packets_cpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyHostToDevice);
    cudaMemcpy(packets_gpu.statuses, packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyHostToDevice);

    // Copy the ciphertext from the PhobosInstance from CPU to GPU.
    std::cout << "Copying the ciphertext from the PhobosInstance on CPU to GPU\n";
    cudaMemcpy(ciphertext_gpu, phobos.ciphertext().data(), 16, cudaMemcpyHostToDevice);

    // Delcare outside the loop to prevent reinitialization
    // Compiler might be doing this under the hood but just in case...
    float percent = 0.0f;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Enter an infinite loop for the brute-force attack.
    while (true)
    {
        // Calculate the percentage of progress and display the current state.
        percent = range->progress() * 100.0;
        std::cout << "\nState: " << range->current() << "/" << range->done_when() << " (" << percent << "%)\n";

        // Record the start time for measuring the duration of each batch.
        t1 = std::chrono::high_resolution_clock::now();

        // Start the SHA task on GPU.
        std::cout << "Starting the SHA task on GPU\n";
        sha_rounds<<<16 * 2048, 512>>>(BATCH_SIZE, packets_gpu.data, packets_gpu.statuses);
        print_cuda_errors();

        // Start the AES task on GPU.
        std::cout << "Starting the AES task on GPU\n";
        aes_decrypt<<<16 * 2048, 512>>>(BATCH_SIZE, packets_gpu.data, packets_gpu.statuses, ciphertext_gpu);
        print_cuda_errors();

        // Wait for CUDA tasks to complete and copy results back to CPU.
        std::cout << "Copying GPU results to CPU\n";
        cudaMemcpy(packets_cpu.data, packets_gpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyDeviceToHost);
        print_cuda_errors();
        cudaMemcpy(packets_cpu.statuses, packets_gpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyDeviceToHost);
        print_cuda_errors();

        // Perform CPU task: Check if the plaintext_cbc matches any packet.
        std::cout << "Waiting for GPU tasks then starting the CPU task to check for match\n";
        if (find_needle(phobos, packets_cpu.data, packets_cpu.statuses, BATCH_SIZE))
        {
            std::cout << "Found needle\n";
            break;
        }

        // Rotate keys and check if the full range is scanned.
        std::cout << "Rotating keys\n";
        if (rotate_keys(range, packets_cpu.data, packets_cpu.statuses, BATCH_SIZE))
        {
            std::cout << "Keyspace exhausted and nothing found\n";
            break;
        }

        std::cout << "CPU task done!\n";

        // Copy the next batch of tasks from CPU to GPU asynchronously.
        cudaMemcpyAsync(packets_gpu.data, packets_cpu.data, BATCH_SIZE * sizeof(Packet), cudaMemcpyHostToDevice);
        print_cuda_errors();
        cudaMemcpyAsync(packets_gpu.statuses, packets_cpu.statuses, BATCH_SIZE * sizeof(PacketStatus), cudaMemcpyHostToDevice);
        print_cuda_errors();

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

    // Free allocated memory on CPU and GPU.
    cudaFree(packets_cpu.data);
    print_cuda_errors();
    cudaFree(packets_gpu.data);
    print_cuda_errors();
    cudaFree(packets_cpu.statuses);
    print_cuda_errors();
    cudaFree(packets_gpu.statuses);
    print_cuda_errors();
}

void showCudaDeviceProp(int device_idx = 0)
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

    showCudaDeviceProp();
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
        brute(phobos, &range);
        return;
    }

    if (std::string(argv[1]) == "show" && std::string(argv[2]) == "config")
    {
        showConfig();
        return 0;
    }

    std::cout << "No, I don't think I will" << std::endl;
    return 2;
}
