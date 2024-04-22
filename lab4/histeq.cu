
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <windows.h> 
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

void salt(Mat image, int n) {
	default_random_engine generater;
	uniform_int_distribution<int>randomRow(0, image.rows - 1);
	uniform_int_distribution<int>randomCol(0, image.cols - 1);

	int i, j;
	for (int k = 0; k < 1000; k++) {
		i = randomRow(generater);
		j = randomCol(generater);
		if (image.channels() == 1) {
			image.at<uchar>(i, j) = 255;
		}
		else if (image.channels() == 3) {
			image.at<Vec3b>(i, j)[0] = 255;
			image.at<Vec3b>(i, j)[1] = 255;
			image.at<Vec3b>(i, j)[2] = 255;
		}
	}
}

__global__ void meanFilter(const uchar* img, uchar* filter, int width, int height) {
	double tmp_f1, tmp_f2, tmp_f3;
	for (int bid = blockDim.y * blockIdx.y + threadIdx.y; bid < height; bid += gridDim.y * blockDim.y)
	{
		if ((bid > 0) && (bid < (height - 1)))
		{
			for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < width; tid += gridDim.x * blockDim.x)
				if ((tid > 0) && (tid < (width - 1)))
				{
					tmp_f1 = 0.0;
					tmp_f2 = 0.0;
					tmp_f3 = 0.0;
					//printf("%d\n", img[(bid - 1) * width + (tid - 1)]);
					tmp_f1 = tmp_f1 + img[((bid - 1) * width + (tid - 1)) * 3 + 0] + img[((bid - 1) * width + (tid)) * 3 + 0] + img[((bid - 1) * width + (tid + 1)) * 3 + 0]
								+ img[((bid)*width + (tid - 1)) * 3 + 0] + img[((bid)*width + (tid)) * 3 + 0] + img[((bid)*width + (tid + 1)) * 3 + 0]
								+ img[((bid + 1) * width + (tid - 1)) * 3 + 0] + img[((bid + 1) * width + (tid)) * 3 + 0] + img[((bid + 1) * width + (tid + 1)) * 3 + 0];
					tmp_f1 /= 9;
					tmp_f2 = tmp_f2 + img[((bid - 1) * width + (tid - 1)) * 3 + 1] + img[((bid - 1) * width + (tid)) * 3 + 1] + img[((bid - 1) * width + (tid + 1)) * 3 + 1]
								+ img[((bid)*width + (tid - 1)) * 3 + 1] + img[((bid)*width + (tid)) * 3 + 1] + img[((bid)*width + (tid + 1)) * 3 + 1]
								+ img[((bid + 1) * width + (tid - 1)) * 3 + 1] + img[((bid + 1) * width + (tid)) * 3 + 1] + img[((bid + 1) * width + (tid + 1)) * 3 + 1];
					tmp_f2 /= 9;
					tmp_f3 = tmp_f3 + img[((bid - 1) * width + (tid - 1)) * 3 + 2] + img[((bid - 1) * width + (tid)) * 3 + 2] + img[((bid - 1) * width + (tid + 1)) * 3 + 2]
								+ img[((bid)*width + (tid - 1)) * 3 + 2] + img[((bid)*width + (tid)) * 3 + 2] + img[((bid)*width + (tid + 1)) * 3 + 2]
								+ img[((bid + 1) * width + (tid - 1)) * 3 + 2] + img[((bid + 1) * width + (tid)) * 3 + 2] + img[((bid + 1) * width + (tid + 1)) * 3 + 2];
					tmp_f3 /= 9;
					filter[(bid * width + tid) * 3 + 0] = (unsigned char)tmp_f1;
					filter[(bid * width + tid) * 3 + 1] = (unsigned char)tmp_f2;
					filter[(bid * width + tid) * 3 + 2] = (unsigned char)tmp_f3;
				}
		}
	}
	/*float tmp_f;
	for (int row = blockDim.y * blockIdx.y + threadIdx.y; row < height; row += gridDim.y * blockDim.y)
		for (int col = blockDim.x * blockIdx.x + threadIdx.x; col < width; col += gridDim.x * blockDim.x)
		{
			filter[(row * width + col) * 3 + 0] = img[(row * width + col) * 3 + 2];
			filter[(row * width + col) * 3 + 1] = img[(row * width + col) * 3 + 1];
			filter[(row * width + col) * 3 + 2] = img[(row * width + col) * 3 + 0];
		}
		*/
}

int main(void) {
	cv::Mat image, image2, outputimg;
	
	image = cv::imread("D:\\opencv\\Lena.png");
	if (image.empty()) {
		cout << "Couldn't open the image" << endl;
		return -1;
	}
	int height = image.rows;
	int width = image.cols;
	int channel = image.channels();
	size_t image_size = sizeof(uchar) * height * width * channel;
	if (!image.isContinuous()) {
		cout << "img1 is not continuous." << endl;
	}
	cv::Mat image1_(height, width, CV_8UC3);
	uchar* d_in = NULL;
	uchar* d_out = NULL;
	cudaMalloc((void**)&d_in, image_size);
	cudaMalloc((void**)&d_out, image_size);
	cudaMemcpy(d_in, image.data, image_size, cudaMemcpyHostToDevice);
	dim3 dimGrid(8, 8, 1);
	dim3 dimBlock(32, 32, 1);
	meanFilter <<<dimGrid, dimBlock>>> (d_in, d_out, width, height);
	cudaMemcpy(image1_.data, d_out, image_size, cudaMemcpyDeviceToHost);
	cv::imshow("image1_", image1_);
	cv::imwrite("D:\\opencv\\Lena2.png", image1_);
	/*image2 = image.clone();
	cv::imshow("Lena", image);
	salt(image, 10000);
	
	cv::imshow("Lena2", image);
	cv::imwrite("D:\\opencv\\Lena.png", image); */
	/*Mat Kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image2, outputimg, image.depth(), Kernel);
	imshow("outputimg", outputimg);*/
	cv::waitKey(0);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0; 
}
