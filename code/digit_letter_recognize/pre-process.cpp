/*************************************************************************
    > File Name: pre-process.cpp
    > Author: li_pengju
    > Mail: li_pengju@vobile.cn 
    > Copyright (c) Vobile Inc. All Rights Reserved
    > Created Time: 2017-09-07 09:52:16
 ************************************************************************/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    Mat src = imread(argv[1], 1);
    cout<<"src.size = "<<src.size()<<endl;
    resize(src, src, Size(0.5*src.cols, 0.5*src.rows));
    Mat gray, blurred, edged;
    cvtColor(src, gray, CV_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5,5), 0, 0,BORDER_DEFAULT );
    Canny(blurred, edged, 50, 200, 3, false );
    imshow("gray", gray);
    imshow("blurred", blurred);
    imshow("edged", edged);
    waitKey();

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( edged, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
    int count = 0;
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( dst, contours, idx, color, CV_FILLED, 8, hierarchy );
        count ++;
    }
    cout<<"Totally contours: "<<count<<endl;
    imshow( "Components", dst );
    waitKey();

    for(idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        cout<<hierarchy[idx]<<endl;
    }
    return 0;
}

