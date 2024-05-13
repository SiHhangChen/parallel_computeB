
#include "./GPULiner.cuh"
#include "./CPULiner.cuh"
#include <windows.h>

using namespace cv;
using namespace std;

void printProgress(
    int step, 
    int total, 
    int self = 0, 
    int cpu = 0) 
{
    // 计算进度
    float progress = (float)step / total;
    int barWidth = 100;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
        }
    std::cout << "] " << int(progress * 100.0) << " %\r";
   
    if(cpu) {
        cout << "cpu";
        if(self) cout << "&&self";
        else cout << "&&opencv";
    }
    else cout << "gpu";
    std::cout.flush();
}

void cannyCPU(
    int self, // 0: use opencv, 1: use self
    int lowThreshold, // double threshold for hysteresis
    int highThreshold, 
    const char* srcImgPath, // path of source image
    const char* dstImgPath, // path of destination image
    int numOfImg) // number of images
{
    if(self){
        for(int i = 0; i < numOfImg; i++){
            string src = srcImgPath + to_string(i) + ".jpg";
            string dst = dstImgPath + to_string(i) + ".jpg";
            canny_c(src, dst, lowThreshold, highThreshold);
            printProgress(i, numOfImg, 1, 1);
        }
    }
    else{
        for(int i = 0; i < numOfImg; i++){
            string src = srcImgPath + to_string(i) + ".jpg";
            string dst = dstImgPath + to_string(i) + ".jpg";
            canny(src, dst, lowThreshold, highThreshold);
            printProgress(i, numOfImg, 0, 1);
        }
    }
}

void cannyGPU(
    int lowThreshold, 
    int highThreshold, 
    const char* srcImgPath, 
    const char* dstImgPath, 
    int numOfImg)
{
    float kernel[9];
    getGaussianKernel__(kernel, 3, 1.0);
    int height, width, channel;
    // int numOfImg = 300;

    for(int i = 0; i < numOfImg; i++){
        Mat image = imread(srcImgPath + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
        if (!image.data) {
            cout << "No image data" << endl;
            return ;
        }
        height = image.rows, width = image.cols, channel = image.channels();
        int imgSize = height * width * channel;
        // cout << "height: " << height << " width: " << width << " channel: " << channel << endl;
        if (!image.isContinuous()) {
            cout << "img1 is not continuous." << endl;
        }
        cv::Mat dstImg(height, width, CV_8UC1);
        uchar *srcArray = NULL, *dstArray = NULL, *gauss = NULL, *grad = NULL, *notMax = NULL;
        short *gradx = NULL, *grady = NULL;
        float *kernelD = NULL;
        cudaMalloc((void**)&srcArray, imgSize * sizeof(uchar));
        cudaMalloc((void**)&dstArray, imgSize * sizeof(uchar));
        cudaMalloc((void**)&gauss, imgSize * sizeof(uchar));
        cudaMalloc((void**)&grad, imgSize * sizeof(uchar));
        cudaMalloc((void**)&notMax, imgSize * sizeof(uchar));
        cudaMalloc((void**)&kernelD, 9 * sizeof(float));
        cudaMalloc((void**)&gradx, imgSize * sizeof(short));
        cudaMalloc((void**)&grady, imgSize * sizeof(short));
        cudaMemcpy(srcArray, image.data, imgSize * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(kernelD, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);
        dim3 dimBlock(8,8);
        dim3 dimGrid((height + dimBlock.y - 1) / dimBlock.y, (width + dimBlock.x - 1) / dimBlock.x);
        // cout << "blockDimX: " << dimBlock.x << " blockDimY: " << dimBlock.y << endl;
        // cout << "gridDimX: " << dimGrid.x << " gridDimY: " << dimGrid.y << endl;
        kernelCanny<<<dimGrid, dimBlock>>>(srcArray, dstArray, gauss, grad, notMax, gradx, grady, width, height, kernelD, 3);
        cudaThreadSynchronize();
        cudaMemcpy(dstImg.data, dstArray, imgSize * sizeof(uchar), cudaMemcpyDeviceToHost);
        imwrite(dstImgPath + to_string(i) + ".jpg", dstImg);
        cudaFree(srcArray);
        cudaFree(dstArray);
        cudaFree(gauss);
        cudaFree(grad);
        cudaFree(notMax);
        cudaFree(kernelD);
        cudaFree(gradx);
        cudaFree(grady);
        printProgress(i, numOfImg, 0, 0);
    }
}

void CANNY(
    int self, 
    int lowThreshold, 
    int highThreshold, 
    const char* srcImgPath, 
    const char* dstImgPath, 
    int numOfImg, 
    int cpu) // 0: use GPU, 1: use CPU
{
    if(cpu){
        cannyCPU(self, lowThreshold, highThreshold, srcImgPath, dstImgPath, numOfImg);
    }
    else{
        cannyGPU(lowThreshold, highThreshold, srcImgPath, dstImgPath, numOfImg);
    }
}