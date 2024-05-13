/*
    the code is used to fix a video's clearness
    time: 2024-5-7
    author: chensihang
*/

#include "INclude.cuh"

using namespace cv;
using namespace std;

//----------------------video to image----------------------
int video2Img(
    const char* videoPath, 
    const char* imgPath) 
{
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return 0;
    }

    Mat frame;
    //把视频转化为图片并保存到指定位置
    int i = 0;
    while (1) {
        cap >> frame;
        if (frame.empty())
            break;
        string name = imgPath + to_string(i) + ".jpg";
        imwrite(name, frame);
        i++;
    }
    return i;
}

//----------------------image to video----------------------
void img2Video(
    const char* imgPath, 
    int numOfImg,
    int width,
    int height,
    const char* videoPath)  
{
    // Change the extension of videoPath to .avi
    std::string videoPathStr(videoPath);
    size_t extensionPos = videoPathStr.rfind('.');
    if (extensionPos != std::string::npos) {
        videoPathStr = videoPathStr.substr(0, extensionPos) + ".avi";
    }

    VideoWriter video(videoPathStr, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(width, height));
    if (!video.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    for (int i = 0; i < numOfImg; i++) {
        string name = imgPath + to_string(i) + ".jpg";
        Mat img = imread(name);
        video.write(img);
    }
}

