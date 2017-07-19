/*************************************************************************
    > File Name: pre_process.cpp
    > Created Time: 2017-07-19 16:59:53
 ************************************************************************/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

const int HEIGHT = 256;
const int WIDTH = 256;
int main(int argc, char* argv[])
{
    if( argc < 2 )
    {
        cout<<"ERROR: inputs error, should be "<<argv[0]<<" image"<<endl;
        return -1;
    }
    Mat srcImage = imread(argv[1], 1);//3 channel color
    Mat dstImage, grayImage;
    resize(srcImage, dstImage, Size(WIDTH, HEIGHT));
    //cvtColor(srcImage, grayImage, CV_BGR2GRAY);
    cout<<"height = "<<dstImage.rows<<"; width = "<<dstImage.cols<<" channel = "<<dstImage.channels()<<endl;
    //imshow("resized", dstImage);
    //waitKey();
    imwrite("dstImage.jpg", dstImage);
    //imwrite("grayImage.jpg", grayImage);
    return 0;
}
