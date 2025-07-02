call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -V
mkdir build
rem nvcc -dc -dlto bruteforce_range.cu main.cu aes256.cu
rem nvcc -dlto bruteforce_range.obj main.obj aes256.obj -O3 -o build\brute.exe
nvcc -lineinfo --ptxas-options=-v bruteforce_range.cu main.cu aes256.cu -O3 -rdc=true --gpu-architecture=native  -o build\brute.exe
rem nvcc -maxrregcount 48 -lineinfo --ptxas-options=-v bruteforce_range.cu main.cu aes256.cu -O3 -rdc=true --gpu-architecture=native  -o build\brute.exe






rem nvcc -lineinfo bruteforce_range.cu main.cu aes256.cu -O3 -rdc=true --gpu-architecture=native  -o build\brute.exe
