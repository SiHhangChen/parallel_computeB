#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>


// Helper functions for CUDA
#include "device_launch_parameters.h"

//#define N 1<<22
#define PI 3.1415926535

class Complex {
public:
    double real;
    double imag;

    Complex() {

    }

    // Wn 获取n次单位复根中的主单位根
    __device__ static Complex W(int n) {
        Complex res = Complex(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

    // Wn^k 获取n次单位复根中的第k个
    __device__ static Complex W(int n, int k) {
        Complex res = Complex(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }

    // 实例化并返回一个复数（只能在Host调用）
    static Complex GetComplex(double real, double imag) {
        Complex r;
        r.real = real;
        r.imag = imag;
        return r;
    }

    // 随机返回一个复数
    static Complex GetRandomComplex() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = (double)rand() / rand();
        return r;
    }

    // 随即返回一个实数
    static Complex GetRandomReal() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = 0;
        return r;
    }

    // 随即返回一个纯虚数
    static Complex GetRandomPureImag() {
        Complex r;
        r.real = 0;
        r.imag = (double)rand() / rand();
        return r;
    }

    // 构造函数（只能在Device上调用）
    __device__ Complex(double real, double imag) {
        this->real = real;
        this->imag = imag;
    }

    // 运算符重载
    __device__ Complex operator+(const Complex& other) {
        Complex res(this->real + other.real, this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex& other) {
        Complex res(this->real - other.real, this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex& other) {
        Complex res(this->real * other.real - this->imag * other.imag, this->imag * other.real + this->real * other.imag);
        return res;
    }
};

// 一维FFT

// 根据数列长度n获取二进制位数, 例如n=8, 则返回3
int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) { // 右移一位
        bits++;
    }
    return bits;
}

// 在二进制位数为bits的前提下求数值i的二进制逆转
__device__ int BinaryReverse(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits; // 逆序累加 i 的二进制位, 当i=2，bits=4; 1011 -> 1101
    } while (i /= 2);
    return r;
}

// 蝴蝶操作, 输出结果直接覆盖原存储单元的数据, factor是旋转因子
__device__ void Butterfly(Complex* a, Complex* b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

// 串行bits
int BinaryReverseS(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits; // 逆序累加 i 的二进制位, 当i=2，bits=4; 1011 -> 1101
    } while (i /= 2);
    return r;
}

void ButterflyS(Complex* a, Complex* b, Complex factor) {
    Complex a1, b1;
    a1.real = a->real + factor.real * b->real - factor.imag * b->imag;
    a1.imag = a->imag + factor.real * b->imag + factor.imag * b->real;
    b1.real = a->real - factor.real * b->real + factor.imag * b->imag;
    b1.imag = a->imag - factor.real * b->imag - factor.imag * b->real;
    *a = a1;
    *b = b1;
}

// FFT算法,对应到GPU的每一个线程，n是数列长度，bits是二进制位数
__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // 线程号，从0开始
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) { // 每次合并的第一个数, 
            int k = i;
            if (n - tid < k) k = n - tid; // 防止越界
            for (int j = 0; j < k / 2; ++j) { // 蝴蝶操作, j是每次合并的第一个数的偏移
                Butterfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, j));//w(k,j)是第j个k次单位复根
            }
        }
        __syncthreads(); // 等待所有线程完成
    }
    result[tid] = nums[BinaryReverse(tid, bits)];
}

// 串行程序，用于验证结果
Complex* FFTSerial(Complex nums[], Complex result[], int n) {
    int bits = GetBits(n);
    for (int i = 0; i < n; ++i) {
        result[BinaryReverseS(i, bits)] = nums[i];
    }
    for (int i = 2; i < 2 * n; i *= 2) {
        for (int j = 0; j < n; j += i) {
            for (int k = 0; k < i / 2; ++k) {
                Complex res;
                res.real = cos(2.0 * PI * k / i);
                res.imag = sin(2.0 * PI * k / i);
                ButterflyS(&result[j + k], &result[j + k + i / 2], res);
            }
        }
    }
    return result;
}

// 打印数列
void printSequence(Complex nums[], const int N) {
    printf("[");
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real, imag = nums[i].imag;
        if (imag == 0) printf("%.16f", real);
        else {
            if (imag > 0) printf("%.16f+%.16fi", real, imag);
            else printf("%.16f%.16fi", real, imag);
        }
        if (i != N - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    srand(time(NULL));  // 设置随机数种子
    const int TPB = 1024;  // 每个Block的线程数，即blockDim.x
    const int N = 1024 * 32;  // 数列大小
    const int bits = GetBits(N); // 二进制位数, 例如N=8, 则bits=3

    // 随机生成实数数列
    Complex* nums = (Complex*)malloc(sizeof(Complex) * N), * dNums, * dResult;
    Complex* nums2 = (Complex*)malloc(sizeof(Complex) * N);
    Complex* sResult = (Complex*)malloc(sizeof(Complex) * N);
    for (int i = 0; i < N; ++i) {
        nums[i] = Complex::GetRandomReal();
        nums2[i] = nums[i];
    }
    printf("Length of Sequence: %d\n", N);
    // printf("Before FFT: \n");
    //printSequence(nums, N);

    // 保存开始时间
    float s = GetTickCount();

    // 分配device内存，拷贝数据到device
    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);

    // 调用kernel
    dim3 threadPerBlock = dim3(TPB); // 一个Block中的线程数, 即blockDim.x
    dim3 blockNum = dim3((N + threadPerBlock.x - 1) / threadPerBlock.x); // Block数目, 即gridDim.x
    FFT << <blockNum, threadPerBlock >> > (dNums, dResult, N, bits); // 调用kernel

    // 拷贝回结果
    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    // 计算用时
    float cost = GetTickCount() - s;
    // printf("After FFT: \n");
    // printSequence(nums, N);
    printf("Time of Transfromation: %fms", cost);
    printf("\n");

    // 释放内存
    
    cudaFree(dNums);
    cudaFree(dResult);

    // 串行结果
    sResult = FFTSerial(nums2, sResult, N);

    // 检查结果
    for (int i = 0; i < N; ++i) {
        if (abs(nums[i].real - sResult[i].real) > 1e-6 || abs(nums[i].imag - sResult[i].imag) > 1e-6) {
            printf("Error!\n");
            break;
        }
    }
    printf("Correct!\n");
    free(nums);
    free(nums2);
    free(sResult);
    return 0;
}
