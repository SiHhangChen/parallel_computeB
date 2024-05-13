#include "INclude.cuh"

using namespace std;
using namespace cv;

void canny(
    string src, 
    string dst, 
    int lowThreshold,
    int highThreshold)
{
    Mat img = imread(src, 0);
    Mat dstImg, edge, gray;
    GaussianBlur(img, img, Size(3,3), 0, 0);
    imwrite("gauss.jpg", img);
    Mat sobel;
    Sobel(img, sobel, CV_16S, 1, 0);
    sobel = 255 - sobel;
    imwrite("sobel.jpg", sobel);
    Canny(img, edge, lowThreshold, highThreshold, 3);
    //颜色反转
    edge = 255 - edge;
    imwrite(dst, edge);
}

//均值超限滤波函数
void meanFilter(
    Mat src, //输入图像
    Mat dst, //输出图像
    int limit) //阈值
{
    int rows = src.rows;
    int cols = src.cols;
    for (int i = 1; i < rows - 1; i++){
        for(int j = 1; j < cols - 1; j++){
            int sum = 0;
            for(int m = -1; m < 2; m++){
                for(int n = -1; n < 2; n++){
                    sum += src.at<uchar>(i+m, j+n);
                }
            }
            sum /= 9;
            if(sum > limit){
                dst.at<uchar>(i, j) = limit;
            }
            else{
                dst.at<uchar>(i, j) = sum;
            }
        }
    }
}

//高斯滤波
void gaussianFilter(
    Mat src, //输入图像
    Mat dst)//输出图像
{
    int rows = src.rows;
    int cols = src.cols;
    Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    for(int i = 1; i < rows - 1; i++){ //边界不处理,可能会出现越界
        for(int j = 1; j < cols - 1; j++){
            float sum = 0;
            for(int m = -1; m < 2; m++){
                for(int n = -1; n < 2; n++){
                    sum += kernel.at<float>(m+1, n+1) * src.at<uchar>(i+m, j+n);
                }
            }
            dst.at<uchar>(i, j) = sum / 16;
        }
    }
}

void calculateGradient(
    Mat src, //输入图像
    Mat &grad,  //输出图像
    Mat &gradx,  //x方向梯度
    Mat &grady) //y方向梯度
{
    gradx = Mat::zeros(src.size(), CV_16SC1);
    grady = Mat::zeros(src.size(), CV_16SC1);
    grad = Mat::zeros(src.size(), CV_8UC1);
    for(int i = 1; i < src.rows - 1; i++){
        for(int j = 1; j < src.cols - 1; j++){
            gradx.at<short>(i, j) = src.at<uchar>(i-1,j+1) - src.at<uchar>(i-1, j-1) + 2*(src.at<uchar>(i, j+1) - src.at<uchar>(i, j-1)) + src.at<uchar>(i+1, j+1) - src.at<uchar>(i+1, j-1);
            grady.at<short>(i, j) = src.at<uchar>(i+1, j-1) - src.at<uchar>(i-1, j-1) + 2*(src.at<uchar>(i+1, j) - src.at<uchar>(i-1, j)) + src.at<uchar>(i+1, j+1) - src.at<uchar>(i-1, j+1);
        }
    }
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            // grad.at<uchar>(i, j) = abs(gradx.at<short>(i, j)) + abs(grady.at<short>(i, j));
            grad.at<uchar>(i, j) = sqrt(gradx.at<short>(i, j) * gradx.at<short>(i, j) + grady.at<short>(i, j) * grady.at<short>(i, j));
        }
    }
}

void nms(
    Mat src, 
    Mat gradx, 
    Mat grady, 
    Mat grad, 
    Mat &dst)
{
    int rows = src.rows;
    int cols = src.cols;
    for(int i = 1; i < rows - 1; i++){
        for(int j = 1; j < cols - 1; j++){
            float dx = gradx.at<short>(i, j);
            float dy = grady.at<short>(i, j);
            if(dx == 0){
                dst.at<uchar>(i, j) = 0;
            }
            else{
                float z = dy / dx;
                //非极大值抑制，这个地方可能出错
                if(z < 1 && z > 0){
                    //梯度线性插值
                    int grad1 = (grad.at<uchar>(i-1, j+1) - grad.at<uchar>(i, j+1)) * abs(z) + grad.at<uchar>(i, j+1);
                    int grad2 = (grad.at<uchar>(i+1, j-1) - grad.at<uchar>(i, j-1)) * abs(z) + grad.at<uchar>(i, j-1);
                    if(grad.at<uchar>(i, j) > grad1 && grad.at<uchar>(i, j) > grad2){
                        dst.at<uchar>(i, j) = grad.at<uchar>(i, j);
                    }
                    else{
                        dst.at<uchar>(i, j) = 0; 
                    }
                }
                else if(z >= 1){
                    int grad1 = (grad.at<uchar>(i-1, j+1) - grad.at<uchar>(i-1, j)) / abs(z) + grad.at<uchar>(i-1, j);
                    int grad2 = (grad.at<uchar>(i+1, j-1) - grad.at<uchar>(i+1, j)) / abs(z) + grad.at<uchar>(i+1, j);
                    if(grad.at<uchar>(i, j) > grad1 && grad.at<uchar>(i, j) > grad2){
                        dst.at<uchar>(i, j) = grad.at<uchar>(i, j);
                    }
                    else{
                        dst.at<uchar>(i, j) = 0;
                    }
                }
                else if(z < 0 && z >= -1){
                    int grad1 = (grad.at<uchar>(i-1, j-1) - grad.at<uchar>(i, j-1)) * abs(z) + grad.at<uchar>(i, j-1);
                    int grad2 = (grad.at<uchar>(i+1, j+1) - grad.at<uchar>(i, j+1)) * abs(z) + grad.at<uchar>(i, j+1);
                    if(grad.at<uchar>(i, j) > grad1 && grad.at<uchar>(i, j) > grad2){
                        dst.at<uchar>(i, j) = grad.at<uchar>(i, j);
                    }
                    else{
                        dst.at<uchar>(i, j) = 0;
                    }
                }
                else{
                    int grad1 = (grad.at<uchar>(i-1, j-1) - grad.at<uchar>(i-1, j)) / abs(z) + grad.at<uchar>(i-1, j);
                    int grad2 = (grad.at<uchar>(i+1, j+1) - grad.at<uchar>(i+1, j)) / abs(z) + grad.at<uchar>(i+1, j);
                    if(grad.at<uchar>(i, j) > grad1 && grad.at<uchar>(i, j) > grad2){
                        dst.at<uchar>(i, j) = grad.at<uchar>(i, j);
                    }
                    else{
                        dst.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }
}

void doubleThreshold(
    Mat src, 
    Mat &dst, 
    int lowThreshold, 
    int highThreshold)
{
    int rows = src.rows;
    int cols = src.cols;
    for(int i = 1; i < rows - 1; i++){
        for(int j = 1; j < cols - 1; j++){
            if(src.at<uchar>(i, j) > highThreshold){
                dst.at<uchar>(i, j) = 0;
            }
            else if(src.at<uchar>(i, j) < lowThreshold){
                dst.at<uchar>(i, j) = 255;
            }
            else{
                if(src.at<uchar>(i-1, j-1) > highThreshold || src.at<uchar>(i-1, j) > highThreshold || src.at<uchar>(i-1, j+1) > highThreshold
                    || src.at<uchar>(i, j-1) > highThreshold || src.at<uchar>(i, j+1) > highThreshold || src.at<uchar>(i+1, j-1) > highThreshold
                    || src.at<uchar>(i+1, j) > highThreshold || src.at<uchar>(i+1, j+1) > highThreshold){
                        dst.at<uchar>(i, j) = 0;
                }
                else{
                    dst.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    // for(int i = 1; i < rows-1; i++){
    //     for(int j = 1; j < cols-1; j++){
    //         if(src.at<uchar>(i-1, j-1) > highThreshold || src.at<uchar>(i-1, j) > highThreshold || src.at<uchar>(i-1, j+1) > highThreshold
    //                 || src.at<uchar>(i, j-1) > highThreshold || src.at<uchar>(i, j+1) > highThreshold || src.at<uchar>(i+1, j-1) > highThreshold
    //                 || src.at<uchar>(i+1, j) > highThreshold || src.at<uchar>(i+1, j+1) > highThreshold){
    //                     dst.at<uchar>(i, j) = 255;
    //         }
    //     }
    // }
}

void canny_c(
    string src, 
    string dst, 
    int lowThreshold, 
    int highThreshold)
{
    Mat img = imread(src, IMREAD_GRAYSCALE);
    Mat gauss, grad, gradx, grady, nonmax, result;
    gauss = Mat::zeros(img.size(), CV_8UC1);
    gaussianFilter(img, gauss);
    // imwrite("gauss.jpg", gauss);
    calculateGradient(gauss, grad, gradx, grady);
    // imwrite("grad.jpg", grad);
    nonmax = Mat::zeros(gauss.size(), CV_8UC1);
    nms(gauss, gradx, grady, grad, nonmax);
    // imwrite("nonmax.jpg", nonmax);
    result = Mat::zeros(gauss.size(), CV_8UC1);
    doubleThreshold(nonmax, result, lowThreshold, highThreshold);
    imwrite(dst, result);
   /* free(gauss);
    free(gradx);
    free(grady);
    free(grad);
    free(nonmax);
    free(result);*/
}

