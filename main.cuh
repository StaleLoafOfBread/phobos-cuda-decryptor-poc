#pragma once
#include "bruteforce_range.cuh"

__device__ void rotate_keys(Packet packets[], PacketStatus statuses[], BruteforceRange *range_gpu);
