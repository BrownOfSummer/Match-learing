/// select_roi_mouse.cpp: 
/// CopyRight (c) 2017 Vobile Inc.  All Rights Reserved.
/// Author: li_pengju  <li_pengju@vobile.cn>
/// Created: 2017-09-07

#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<string>
#include<cstdio>
using namespace cv;
using namespace std;

Point point1, point2; /* vertical points of the bounding box */
int drag = 0; /* move with button down or not */
int select_flag = 0; /* flag the rectangle is select or not*/
Rect rect; /* bounding box */
Mat img, roiImg; /* roiImg - the part of the image in the bounding box */


void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
                            
    }
         
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        Mat img1 = img.clone();
        point2 = Point(x, y);
        rectangle(img1, point1, point2, CV_RGB(0, 255, 0), 1, 8, 0);
        imshow("image", img1);
                                                
    }
             
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        rect = Rect(point1.x,point1.y,x-point1.x,y-point1.y);
        drag = 0;
        roiImg = img(rect);
                                                    
    }
                 
    if (event == CV_EVENT_LBUTTONUP)
    {
        /* ROI selected */
        select_flag = 1;
        drag = 0;
                                        
    }
}
void help()
{
    cout << "\nThis program demonstrated how to select several rois in a image and save to image\n"
            "Call:\n"
            "./select_roi_mouse [image_name]\n" << endl;
    cout << "\nSelected area will draw red rectangle on src image, green rectangle for confirm"<<endl;

    cout << "Hot keys: \n"
            "\tESC - quit the program, and save the selected images\n"
            "\ts - is the selected area is ok, press s save to vectors\n"
            "\tn - do next select operation\n";
}
int main(int argc, char *argv[])
{
    vector<Mat> rois;
    string out_name;
    char tmp[200];
    img = imread(argv[1], 1);
    if( img.empty() ) {
        cout<< "Image empty !"<<endl;
        return -1;
    }
    //help();
    imshow("image", img);
    cvSetMouseCallback("image", mouseHandler, NULL);
    for(;;)
    {
        //cvSetMouseCallback("image", mouseHandler, NULL);
        if (select_flag == 1)
        {
            imshow("ROI", roiImg); /* show the image bounded by the box */
        }
        else {
            cout<< "Select failed, press 'n' to re-select or 'esc' to quit !"<<endl;
        }
        char c = (char)waitKey(0);
        if (c == 27)
        {
            cout<<"Exiting ..."<<endl;
            cout<<"Selected "<<rois.size()<< " rois."<<endl;
            if(rois.size() > 0){
                cout<<"Write rois to images !"<<endl;
                for(int i = 0; i < rois.size(); i ++){
                    sprintf(tmp, "roi%d.jpg",i);
                    out_name = tmp;
                    imwrite(out_name, rois[i]);
                }
            }
            break;
        }
        switch( c )
        {
        case 's':
            {
                cout<<"Save roi to vector ...."<<endl;
                rois.push_back(roiImg.clone());
                rectangle(img, rect, CV_RGB(255, 0, 0), 1, 8, 0);
                imshow("image", img);
            }
            break;
        case 'n':
            {
                cout<<"Do next select ...."<<endl;
                select_flag = 0;
            }
            break;
        }
                                                                                    
    }
    return 0;
}
