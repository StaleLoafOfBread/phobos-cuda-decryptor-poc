// https://cert.pl/en/posts/2023/02/breaking-phobos/
// https://cert.pl/uploads/2023/02/breaking-phobos/key_sched.png
#include <windows.h>

void key_sched()
{
    struct _SYSTEMTIME SystemTime;  // [esp+8h] [ebp-20h] BYREF
    struct _FILETIME FileTime;      // [esp+18h] [ebp-10h] BYREF
    LARGE_INTEGER PerformanceCount; // [esp+20h] [ebp-8hJ BYREF
    InitializeCriticalSectionAndSpinCount(&CriticalSection, OxFAOu);
    EnterCriticalSection(&CriticalSection);
    QueryPerformanceCounter(&PerformanceCount);
    kk0 ^= GetTickCount();
    kk1 ^= PerformanceCount.HighPart;
    kk2 ^= PerformanceCount.LowPart;
    kk3 ^= GetCurrentProcessId();
    kk4 ^= GetCurrentThreadId();
    GetLocalTime(&SystemTime);
    SystemTimeToFileTime(&SystemTime, &FileTime);
    kk5 ^= FileTime.dwHighDateTime;
    kk6 ^= FileTime.dwLowDateTime;
    QueryPerformanceCounter(&PerformanceCount);
    kk7 ^= PerformanceCount.HighPart;
    kk0 ^= PerformanceCount.LowPart;
    do
        key_rotate_round();
    while ((_BYTE)kkO);
    key_init = 1;
    LeaveCriticalSection(&CriticalSection);
}

// kk0
// kk1
// kk2
// kk3
// kk4
// kk5
// kk6
// kk7

// pid
// tid
// tick
// filetime
// percounter
// percounter
