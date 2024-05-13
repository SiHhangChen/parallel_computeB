#include "./INclude.cuh"

using namespace cv;
using namespace std;

//----------------------CUDA-OF-IMAGE----------------------
void getGaussianKernel__(
    float* kernel, 
    int size, 
    float sigma)
{
    float sum = 0;
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            kernel[i * size + j] = exp(-((i - size / 2) * (i - size / 2) + (j - size / 2) * (j - size / 2)) / (2 * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            kernel[i * size + j] /= sum;
        }
    }
}

__device__ void gaussianFilter(
    uchar* srcImg, 
    uchar* dstImg, 
    int width, 
    int height, 
    int idx, 
    int idy, 
    float* kernel, 
    int size)
{
    // int kerneldd[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    // dstImg[idx * width + idy] = srcImg[idx * width + idy];
    int sum = 0;
    for(int i = -1; i < 2; i++){
        for(int j = -1; j < 2; j++){
            int x = idx + i;
            int y = idy + j;
            if(x < 0) x = 0;
            if(x >= height) x = height - 1;
            if(y < 0) y = 0;
            if(y >= width) y = width - 1;
            sum += srcImg[(idx + i) * width + idy + j] * kernel[(i + 1) * size + j + 1];
        }
    }
    dstImg[idx * width + idy] = sum;
}

__device__ void calculateGradient(
    uchar* srcImg, 
    uchar* grad, 
    short* gradx, 
    short* grady, 
    int width, 
    int idx, 
    int idy)
{
    gradx[idx * width + idy] = srcImg[(idx - 1) * width + idy + 1] + 2 * srcImg[idx * width + idy + 1] + srcImg[(idx + 1) * width + idy + 1] 
                                - srcImg[(idx - 1) * width + idy - 1] - 2 * srcImg[idx * width + idy - 1] - srcImg[(idx + 1) * width + idy - 1];
    grady[idx * width + idy] = srcImg[(idx + 1) * width + idy - 1] + 2 * srcImg[(idx + 1) * width + idy] + srcImg[(idx + 1) * width + idy + 1] 
                                - srcImg[(idx - 1) * width + idy - 1] - 2 * srcImg[(idx - 1) * width + idy] - srcImg[(idx - 1) * width + idy + 1];
    //printf("gradx: %d, grady: %d\n", gradx[idx * width + idy], grady[idx * width + idy]);
    grad[idx * width + idy] = abs(gradx[idx * width + idy]) + abs(grady[idx * width + idy]);
}

//比较梯度值
__device__ int getgrad(
    uchar grad, 
    uchar grad1, 
    uchar grad2)
{
    if(grad > grad1 && grad > grad2){
        return grad;
    }
    else{
        return 0;
    }
}

__device__ void nms(
    uchar* dstImg, //输出图像
    uchar* grad, //梯度图像
    short* gradx, //x方向梯度
    short* grady, //y方向梯度
    int width, 
    int idx, 
    int idy)
{
    float dx = gradx[idx * width + idy];
    float dy = grady[idx * width + idy];
    if(dx = 0){
        dstImg[idx * width + idy] = 0;
    }
    else{
        uchar grad1, grad2;
        float z = dy/dx;
        if(z < 1 && z > 0){
            //梯度线性插值
            grad1 = (grad[(idx-1) * width + idy + 1] - grad[idx * width + idy + 1]) * abs(z) + grad[idx * width + idy + 1];
            grad2 = (grad[(idx+1) * width + idy - 1] - grad[idx * width + idy - 1]) * abs(z) + grad[idx * width + idy - 1];
            dstImg[idx * width + idy] = getgrad(grad[idx * width + idy], grad1, grad2);
        }
        else if(z >= 1){
            grad1 = (grad[(idx-1) * width + idy + 1] - grad[(idx - 1) * width + idy]) / abs(z) + grad[(idx - 1) * width + idy];
            grad2 = (grad[(idx+1) * width + idy - 1] - grad[(idx + 1) * width + idy]) / abs(z) + grad[(idx + 1) * width + idy];
            dstImg[idx * width + idy] = getgrad(grad[idx * width + idy], grad1, grad2);
        }
        else if(z < 0 && z >= -1){
            grad1 = (grad[(idx-1) * width + idy - 1] - grad[idx * width + idy - 1]) / abs(z) + grad[idx * width + idy - 1];
            grad2 = (grad[(idx+1) * width + idy + 1] - grad[idx * width + idy + 1]) / abs(z) + grad[idx * width + idy + 1];
            dstImg[idx * width + idy] = getgrad(grad[idx * width + idy], grad1, grad2);
        }
        else{
            grad1 = (grad[(idx-1) * width + idy - 1] - grad[(idx - 1) * width + idy]) / abs(z) + grad[(idx - 1) * width + idy];
            grad2 = (grad[(idx+1) * width + idy + 1] - grad[(idx + 1) * width + idy]) / abs(z) + grad[(idx + 1) * width + idy];
            dstImg[idx * width + idy] = getgrad(grad[idx * width + idy], grad1, grad2);
        }
    }
}

__device__ void doubleThreshold(
    uchar* dstImg, 
    uchar* grad, 
    int width, 
    int idx, 
    int idy, 
    int low, 
    int high)
{
    if(grad[idx * width + idy] > high){
        dstImg[idx * width + idy] = 0;
    }
    else if(grad[idx * width + idy] < low){
        dstImg[idx * width + idy] = 255;
    }
    else{
        if(grad[(idx-1) * width + idy - 1] > high || grad[(idx-1) * width + idy] > high || grad[(idx-1) * width + idy + 1] > high 
                || grad[idx * width + idy - 1] > high || grad[idx * width + idy + 1] > high || grad[(idx+1) * width + idy - 1] > high 
                || grad[(idx+1) * width + idy] > high || grad[(idx+1) * width + idy + 1] > high){
            dstImg[idx * width + idy] = 0;
        }
        else{
            dstImg[idx * width + idy] = 255;
        }
    }

    //左右上下，对角线连接
    // if(grad[(idx-1) * width + idy - 1] > high && grad[(idx+1) * width + idy + 1] > high){
    //     dstImg[idx * width + idy] = 0;
    // }
    // else if(grad[(idx-1) * width + idy + 1] > high && grad[(idx+1) * width + idy - 1] > high){
    //     dstImg[idx * width + idy] = 0;
    // }
    // else if(grad[(idx-1) * width + idy] > high && grad[(idx+1) * width + idy] > high){
    //     dstImg[idx * width + idy] = 0;
    // }
    // else if(grad[idx * width + idy - 1] > high && grad[idx * width + idy + 1] > high){
    //     dstImg[idx * width + idy] = 0;
    // }

    // if(grad[(idx-1) * width + idy - 1] > high || grad[(idx-1) * width + idy] > high || grad[(idx-1) * width + idy + 1] > high 
    //             || grad[idx * width + idy - 1] > high || grad[idx * width + idy + 1] > high || grad[(idx+1) * width + idy - 1] > high 
    //             || grad[(idx+1) * width + idy] > high || grad[(idx+1) * width + idy + 1] > high){
    //         dstImg[idx * width + idy] = 0;
    // }
}   


__global__ void kernelCanny(
    uchar* srcImg, //输入图像
    uchar* dstImg, //输出图像
    uchar* gauss, //高斯滤波后的图像
    uchar* grad, //梯度图像
    uchar* notMax, //非极大值抑制后的图像
    short* gradx, //x方向梯度
    short* grady, //y方向梯度
    int width, //图像宽度
    int height, //图像高度
    float* kernel, //高斯核
    int size)//核大小
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < height - 1 && idy < width - 1 && idx > 0 && idy > 0){
        gaussianFilter(srcImg, dstImg, width, height, idx, idy, kernel, size);
        srcImg[idx * width + idy] = dstImg[idx * width + idy];
        __syncthreads();
        calculateGradient(srcImg, dstImg, gradx, grady, width, idx, idy);
        grad[idx * width + idy] = dstImg[idx * width + idy];
        __syncthreads();
        nms(dstImg, grad, gradx, grady, width, idx, idy);
        notMax[idx * width + idy] = dstImg[idx * width + idy];
        __syncthreads();
        doubleThreshold(dstImg, notMax, width, idx, idy, 50, 100);
        __syncthreads();
    }
}