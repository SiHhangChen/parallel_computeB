/*
    the code is used to fix a video's clearness
    time: 2024-5-7
    author: chensihang
*/
#include "./video2Img.cuh"
#include "./CANNY.cuh"

using namespace cv;
using namespace std;

int main() {
    int self = 0;
    int lowThreshold = 50;
    int highThreshold = 100;
    int cpu = 0;
    cout << "Please input self and cpu" << endl;
    cin >> self >> cpu;
    int startTime = GetTickCount();
    int numOfImg = video2Img("../example.mp4", "../imgs/");
    const char* srcImgPath = "../imgs/";
    const char* dstImgPath = "../dstimgs/";
    CANNY(self, lowThreshold, highThreshold, srcImgPath, dstImgPath, numOfImg, cpu);
    int width = imread("../imgs/0.jpg").cols;
    int height = imread("../imgs/0.jpg").rows;
    img2Video(dstImgPath, numOfImg, width, height, "../dst.avi");
    int endTime = GetTickCount();
    cout << "Time: " << endTime - startTime << "ms" << endl;
    return 0;
}