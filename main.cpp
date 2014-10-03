//
//  main.cpp
//  SecondOpencvApp
//
//  Created by Yu Wang on 9/15/14.
//  Copyright (c) 2014 Yu. All rights reserved.
//

#include <iostream>
#include <array>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "tools.h"
using namespace std;
using namespace cv;


/** global variables **/
int mouse_count = 0;
Point corner_points[20];//-> single step
//Point corner_points[8] -> 2-step method
Point *fp2, *wp;
/************* H matrix *************/
//Mat H_mat(3,3,CV_64FC1);
/*********** Images ************/
Mat img, img2;
//display matrix
void disp_mat(Mat disp_mat){
    
    int rnum = disp_mat.rows;
    int cnum = disp_mat.cols;
    cout << "rows: " << rnum <<"cols: "<< cnum<< endl;
    
    for (int i = 0; i < rnum; i++) {
        for (int j = 0; j < cnum; j++) {
            cout << "|" << disp_mat.at<double>(i,j);
        }
        cout << endl;
    }
    
};
//calculate vanishing point
Mat vpoint(Point *line1_pts, Point *line2_pts){
    Mat vp_result(3,1,CV_64FC1);

    Mat hm_l1pt1 = (Mat_<double>(3,1)<< line1_pts[0].x, line1_pts[0].y, 1);
    Mat hm_l1pt2 = (Mat_<double>(3,1)<< line1_pts[1].x, line1_pts[1].y, 1);
    Mat hm_l2pt1 = (Mat_<double>(3,1)<< line2_pts[0].x, line2_pts[0].y, 1);
    Mat hm_l2pt2 = (Mat_<double>(3,1)<< line2_pts[1].x, line2_pts[1].y, 1);
    
    vp_result = (hm_l2pt1.cross(hm_l2pt2)).cross(hm_l1pt1.cross(hm_l1pt2));
    cout << "vanishing points:" << vp_result << endl;
    return vp_result;
};
//calculate vanishing line
Mat vline(Mat vl_pt1, Mat vl_pt2){
    Mat temp(3,1,CV_64FC1);
    temp = vl_pt1.cross(vl_pt2);
    cout << temp << endl;
    cout << "x1: " << temp.at<double>(0,0) << endl;
    cout << "x2: " << temp.at<double>(1,0)<< endl;
    cout << "x3: " << temp.at<double>(2,0)<< endl;
    
    
    temp.at<double>(0,0) = temp.at<double>(0,0)/temp.at<double>(2,0);

    temp.at<double>(1,0) = temp.at<double>(1,0)/temp.at<double>(2,0);
    
    temp.at<double>(2,0) = 1;
    cout << temp << endl;
    return temp;
};
//esimate matrix S for 2-step method
Mat calc_S(Point *orth_points){
    Mat line1(3,1,CV_64FC1);
    Mat line2(3,1,CV_64FC1);
    Mat line3(3,1,CV_64FC1);
    Mat line4(3,1,CV_64FC1);
    // points
    Mat hm_l1pt1 = (Mat_<double>(3,1)<< orth_points[0].x, orth_points[0].y, 1);
    Mat hm_l1pt2 = (Mat_<double>(3,1)<< orth_points[1].x, orth_points[1].y, 1);
    Mat hm_l2pt1 = (Mat_<double>(3,1)<< orth_points[2].x, orth_points[2].y, 1);
    Mat hm_l2pt2 = (Mat_<double>(3,1)<< orth_points[3].x, orth_points[3].y, 1);
    Mat hm_l3pt1 = (Mat_<double>(3,1)<< orth_points[4].x, orth_points[4].y, 1);
    Mat hm_l3pt2 = (Mat_<double>(3,1)<< orth_points[5].x, orth_points[5].y, 1);
    Mat hm_l4pt1 = (Mat_<double>(3,1)<< orth_points[6].x, orth_points[6].y, 1);
    Mat hm_l4pt2 = (Mat_<double>(3,1)<< orth_points[7].x, orth_points[7].y, 1);
    
    //lines
    line1 = hm_l1pt1.cross(hm_l1pt2);
    line2 = hm_l2pt1.cross(hm_l2pt2);
    line3 = hm_l3pt1.cross(hm_l3pt2);
    line4 = hm_l4pt1.cross(hm_l4pt2);
    

    //normalize
    double l11 = line1.at<double>(0, 0)/line1.at<double>(2, 0);
    double l12 = line1.at<double>(1, 0)/line1.at<double>(2, 0);
    double l21 = line2.at<double>(0, 0)/line2.at<double>(2, 0);
    double l22 = line2.at<double>(1, 0)/line2.at<double>(2, 0);
    double l31 = line3.at<double>(0, 0)/line3.at<double>(2, 0);
    double l32 = line3.at<double>(1, 0)/line3.at<double>(2, 0);
    double l41 = line4.at<double>(0, 0)/line4.at<double>(2, 0);
    double l42 = line4.at<double>(1, 0)/line4.at<double>(2, 0);
    //system: sys_mat * S = sys_b
    Mat sys_mat = (Mat_<double>(2,2)<< l11*l21,l11*l22+l12*l21,l31*l41,l31*l42+l32*l41);
    Mat sys_b = (Mat_<double>(2,1)<< -l12*l22,-l32*l42);
    
    Mat temp = sys_mat.inv()*sys_b;
    Mat S_mat = (Mat_<double>(2,2)<< temp.at<double>(0,0),temp.at<double>(1,0),temp.at<double>(1,0),1);
    
    return S_mat;
};
//esimate matrix H for single step method
Mat calc_H_one_step(Point *orth_points){
    Mat line1(3,1,CV_64FC1);
    Mat line2(3,1,CV_64FC1);
    Mat line3(3,1,CV_64FC1);
    Mat line4(3,1,CV_64FC1);
    Mat line5(3,1,CV_64FC1);
    Mat line6(3,1,CV_64FC1);
    Mat line7(3,1,CV_64FC1);
    Mat line8(3,1,CV_64FC1);
    Mat line9(3,1,CV_64FC1);
    Mat line10(3,1,CV_64FC1);
    // points
    Mat hm_l1pt1 = (Mat_<double>(3,1)<< orth_points[0].x, orth_points[0].y, 1);
    Mat hm_l1pt2 = (Mat_<double>(3,1)<< orth_points[1].x, orth_points[1].y, 1);
    Mat hm_l2pt1 = (Mat_<double>(3,1)<< orth_points[2].x, orth_points[2].y, 1);
    Mat hm_l2pt2 = (Mat_<double>(3,1)<< orth_points[3].x, orth_points[3].y, 1);
    Mat hm_l3pt1 = (Mat_<double>(3,1)<< orth_points[4].x, orth_points[4].y, 1);
    Mat hm_l3pt2 = (Mat_<double>(3,1)<< orth_points[5].x, orth_points[5].y, 1);
    Mat hm_l4pt1 = (Mat_<double>(3,1)<< orth_points[6].x, orth_points[6].y, 1);
    Mat hm_l4pt2 = (Mat_<double>(3,1)<< orth_points[7].x, orth_points[7].y, 1);
    Mat hm_l5pt1 = (Mat_<double>(3,1)<< orth_points[8].x, orth_points[8].y, 1);
    Mat hm_l5pt2 = (Mat_<double>(3,1)<< orth_points[9].x, orth_points[9].y, 1);
    Mat hm_l6pt1 = (Mat_<double>(3,1)<< orth_points[10].x, orth_points[10].y, 1);
    Mat hm_l6pt2 = (Mat_<double>(3,1)<< orth_points[11].x, orth_points[11].y, 1);
    Mat hm_l7pt1 = (Mat_<double>(3,1)<< orth_points[12].x, orth_points[12].y, 1);
    Mat hm_l7pt2 = (Mat_<double>(3,1)<< orth_points[13].x, orth_points[13].y, 1);
    Mat hm_l8pt1 = (Mat_<double>(3,1)<< orth_points[14].x, orth_points[14].y, 1);
    Mat hm_l8pt2 = (Mat_<double>(3,1)<< orth_points[15].x, orth_points[15].y, 1);
    Mat hm_l9pt1 = (Mat_<double>(3,1)<< orth_points[16].x, orth_points[16].y, 1);
    Mat hm_l9pt2 = (Mat_<double>(3,1)<< orth_points[17].x, orth_points[17].y, 1);
    Mat hm_l10pt1 = (Mat_<double>(3,1)<< orth_points[18].x, orth_points[18].y, 1);
    Mat hm_l10pt2 = (Mat_<double>(3,1)<< orth_points[19].x, orth_points[19].y, 1);
    // lines
    line1 = hm_l1pt1.cross(hm_l1pt2);
    line2 = hm_l2pt1.cross(hm_l2pt2);
    line3 = hm_l3pt1.cross(hm_l3pt2);
    line4 = hm_l4pt1.cross(hm_l4pt2);
    line5 = hm_l5pt1.cross(hm_l5pt2);
    line6 = hm_l6pt1.cross(hm_l6pt2);
    line7 = hm_l7pt1.cross(hm_l7pt2);
    line8 = hm_l8pt1.cross(hm_l8pt2);
    line9 = hm_l9pt1.cross(hm_l9pt2);
    line10 = hm_l10pt1.cross(hm_l10pt2);
    
    //normalize
    double l11 = line1.at<double>(0, 0)/line1.at<double>(2, 0);
    double l12 = line1.at<double>(1, 0)/line1.at<double>(2, 0);
    double l21 = line2.at<double>(0, 0)/line2.at<double>(2, 0);
    double l22 = line2.at<double>(1, 0)/line2.at<double>(2, 0);
    double l31 = line3.at<double>(0, 0)/line3.at<double>(2, 0);
    double l32 = line3.at<double>(1, 0)/line3.at<double>(2, 0);
    double l41 = line4.at<double>(0, 0)/line4.at<double>(2, 0);
    double l42 = line4.at<double>(1, 0)/line4.at<double>(2, 0);
    double l51 = line5.at<double>(0, 0)/line5.at<double>(2, 0);
    double l52 = line5.at<double>(1, 0)/line5.at<double>(2, 0);
    double l61 = line6.at<double>(0, 0)/line6.at<double>(2, 0);
    double l62 = line6.at<double>(1, 0)/line6.at<double>(2, 0);
    double l71 = line7.at<double>(0, 0)/line7.at<double>(2, 0);
    double l72 = line7.at<double>(1, 0)/line7.at<double>(2, 0);
    double l81 = line8.at<double>(0, 0)/line8.at<double>(2, 0);
    double l82 = line8.at<double>(1, 0)/line8.at<double>(2, 0);
    double l91 = line9.at<double>(0, 0)/line9.at<double>(2, 0);
    double l92 = line9.at<double>(1, 0)/line9.at<double>(2, 0);
    double l101 = line10.at<double>(0, 0)/line10.at<double>(2, 0);
    double l102 = line10.at<double>(1, 0)/line10.at<double>(2, 0);
    
    //system: sys_mat * S = sys_b
    Mat sys_mat = (Mat_<double>(5,5)<< \
       l11*l21,(l11*l22+l12*l21)/2,l12*l22,(l11+l21)/2,(l12+l22)/2, \
       l31*l41,(l31*l42+l32*l41)/2,l32*l42,(l31+l41)/2,(l32+l42)/2, \
       l51*l61,(l51*l62+l52*l61)/2,l52*l62,(l51+l61)/2,(l52+l62)/2, \
       l71*l81,(l71*l82+l72*l81)/2,l72*l82,(l71+l81)/2,(l72+l82)/2, \
       l91*l101,(l91*l102+l92*l101)/2,l92*l102,(l91+l101)/2,(l92+l102)/2);
    cout << "system: "<< sys_mat << endl;
    Mat sys_b = (Mat_<double>(5,1)<< -1,-1,-1,-1,-1);
    
    Mat temp = sys_mat.inv()*sys_b;
    Mat S_mat = (Mat_<double>(2,2)<< temp.at<double>(0,0),temp.at<double>(1,0)/2,\
                                     temp.at<double>(1,0)/2,temp.at<double>(2,0));
    Mat g = (Mat_<double>(2,1)<< temp.at<double>(3,0)/2,temp.at<double>(4,0)/2);
    Mat U,D2,D,Ut;
    SVD::compute(S_mat, D2, U, Ut,0);
    pow(D2, 0.5, D);
    D = Mat::diag(D);
    Mat A = U * D * U.inv();
    //cout << "S:" << S_mat<< endl;
    //cout << "A:" << A << endl;
    Mat v = A.inv()*g;
    //cout << "v:" << v << endl;
    Mat C_mat = (Mat_<double>(3,3) << \
        0.001*A.at<double>(0,0)/A.at<double>(1,1),0.001*A.at<double>(0,1)/A.at<double>(1,1),0,\
        0.001*A.at<double>(1.0)/A.at<double>(1,1),0.001*A.at<double>(1,1)/A.at<double>(1,1),0,\
        0.001*v.at<double>(0,0)/A.at<double>(1,1),0.001*v.at<double>(1,0)/A.at<double>(1,1),\
        1/A.at<double>(1,1) );
    return C_mat;
};
// build affine matrix from K and v
Mat build_affine_mat(Mat mat_S){
  
    Mat U,D2,D,Ut;
    SVD::compute(mat_S, D2, U, Ut,0);
    pow(D2, 0.5, D);
    D = Mat::diag(D);
    Mat A = U * D * U.inv();
    Mat affine_mat = (Mat_<double>(3,3)<< \
        A.at<double>(0,0), A.at<double>(0,1),0, A.at<double>(1,0), A.at<double>(1,1),0,0,0,1);
    return affine_mat;
}
/* projective correction matrix (PCM)  NOT used*/
void createPCM(Mat vl, Mat PCM){
    PCM = (Mat_<double>(3,3)<< \
        1,0,0,0,1,0, vl.at<double>(0,0), vl.at<double>(1,0), vl.at<double>(2,0));
    
};

/* Calculate the bounding box from 4 points */
long* search_boundary_x(Point *points){
    long max = 0;
    long min = 3000;//assume the maximum resolution
    long bound[2];
    long *bound_p = bound;
    for (int i = 0; i< 4; i++) {
        if (points[i].x > max) {
            max = points[i].x;
        }
        if (points[i].x < min) {
            min = points[i].x;
        }
    }
    bound[0] = min;
    bound[1] = max;
    return bound_p;
}
long* search_boundary_y(Point *points){
    long max = 0;
    long min = 3000;//assume the maximum resolution
    long bound[2];
    long *bound_p = bound;
    for (int i = 0; i< 4; i++) {
        if (points[i].y > max) {
            max = points[i].y;
        }
        if (points[i].y < min) {
            min = points[i].y;
        }
    }
    bound[0] = min;
    bound[1] = max;
    return bound_p;
}
/* Backward search (task 1) */
void back_search(Point *origin_points, Point *target_points, Mat origin, Mat target, Mat H_mat){
    Mat temp(3,1,CV_64FC1);
    Mat result(3,1,CV_64FC1);
    long ox_min, ox_max, oy_min, oy_max;
    ox_min = search_boundary_x(origin_points)[0];
    ox_max = search_boundary_x(origin_points)[1];
    oy_min = search_boundary_y(origin_points)[0];
    oy_max = search_boundary_y(origin_points)[1];
    
    long tx_min, tx_max, ty_min, ty_max;
    tx_min = search_boundary_x(target_points)[0];
    tx_max = search_boundary_x(target_points)[1];
    ty_min = search_boundary_y(target_points)[0];
    ty_max = search_boundary_y(target_points)[1];
    
    for (long i = ox_min; i < ox_max; i++) {
        for (long j = oy_min; j < oy_max; j++) {
            temp.at<double>(0,0)=i;
            temp.at<double>(1,0)=j;
            temp.at<double>(2,0)=1;
            result = H_mat.inv()*temp;
            
            long x = result.at<double>(0,0)/result.at<double>(2,0);
            long y = result.at<double>(1,0)/result.at<double>(2,0);
            //cout<< "x: " << x << "y: "<< y << endl;
            if (x > tx_min && x < tx_max && y > ty_min && y < ty_max) {
                
                origin.at<Vec3b>(j,i) = target.at<Vec3b>(y,x);
            }
            /*for testing: origin.at<Vec3b>(j,i) = Vec3b(1,1,1);*/
        }
    }
    try {
        imwrite("image_overlap.png", origin);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
}
/* Backward search (task 2) */
void back_search_task2(Point *origin_points, Mat img, Mat H_matrix){
    cout << "hello!"; 
    long tx_min, tx_max, ty_min, ty_max;
    Mat temp(3,1,CV_64FC1);
    Mat result(3,1,CV_64FC1);
    Mat H_mat(H_matrix.size(),H_matrix.type());
    H_matrix.copyTo(H_mat);
    Point bountary_points[4];
    long ox_min, ox_max, oy_min, oy_max;
    cout << origin_points[0] << endl;
    cout << origin_points[1] << endl;
    cout << origin_points[2] << endl;
    cout << origin_points[3] << endl;
    ox_min = search_boundary_x(origin_points)[0];
    ox_max = search_boundary_x(origin_points)[1];
    oy_min = search_boundary_y(origin_points)[0];
    oy_max = search_boundary_y(origin_points)[1];
    
    for (int i = 0; i < 4; i++) {
        temp.at<double>(0,0) = origin_points[i].x;
        temp.at<double>(1,0) = origin_points[i].y;
        temp.at<double>(2,0) = 1;
        result = H_mat.inv()*temp;
        long x = result.at<double>(0,0)/result.at<double>(2,0);
        long y = result.at<double>(1,0)/result.at<double>(2,0);
        bountary_points[i].x = x;
        bountary_points[i].y = y;
    }
    disp_mat(H_mat);
    cout<< bountary_points[0]<< endl;
    cout<< bountary_points[1]<< endl;
    cout<< bountary_points[2]<< endl;
    cout<< bountary_points[3]<< endl;
    tx_min = search_boundary_x(bountary_points)[0];
    tx_max = search_boundary_x(bountary_points)[1];
    ty_min = search_boundary_y(bountary_points)[0];
    ty_max = search_boundary_y(bountary_points)[1];
    
    Mat img_rect(ty_max-ty_min, tx_max-tx_min, CV_8UC3, Scalar(0,0,0));
    
    long xx,yy;
    cout << img_rect.rows << "::" << img_rect.cols << endl;
    for (long i = tx_min; i < tx_max; i++) {
        for (long j = ty_min; j < ty_max; j++) {
            temp.at<double>(0,0)=i;
            temp.at<double>(1,0)=j;
            temp.at<double>(2,0)=1;
            result = H_mat*temp;
            
            xx = result.at<double>(0,0)/result.at<double>(2,0);
            yy = result.at<double>(1,0)/result.at<double>(2,0);
            //cout<< "x: " << x << "y: "<< y << endl; // for testing
            if (xx > ox_min && xx < ox_max && yy > oy_min && yy < oy_max) {
                img_rect.at<Vec3b>(j - ty_min, i - tx_min) = img.at<Vec3b>(yy,xx);
                //cout<< "x: " << xx << "   y: "<< yy << endl;
            }else{
                //fill the blank space with Blue
                img_rect.at<Vec3b>(j - ty_min, i - tx_min) = Vec3b(100,0,0);
            }
            //origin.at<Vec3b>(j,i) = Vec3b(1,1,1); // for testing
        }
    }
    Mat output;
    cout << "size:: "<< img_rect.size()<< endl;
    // interpolation can be performed HERE
    cv::resize(img_rect, output, Size(), 1, 1, INTER_CUBIC);

    try {
        imwrite("img_step2.png", output);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    
}
/* Mouse callback: click to get the coordinate values */
void callBackFunc(int event, int x, int y, int flags, void* userdata){
    //upper limit: 4
    Point *p = (Point*) userdata;
    
    if (event == EVENT_LBUTTONDOWN) {
        cout << "Left button clicked at (" << x << "," << y << ")" << endl;
        p[mouse_count].x = x;
        p[mouse_count].y = y;
        
        if (mouse_count%2 == 1) {
            drawLine(img, p[mouse_count-1], p[mouse_count]);
            imshow("testImage", img);

        }
        mouse_count++;
        cout<<"count: "<<mouse_count<<endl;
    }
    // SHOULD comment other step to run desired step

    //STEP 1
    if (mouse_count >= 8){
        // unregister listener
        
        setMouseCallback("testImage", NULL, NULL);
     
        // 2-step method
        Mat vp1 = vpoint(corner_points, corner_points+2);
        Mat vp2 = vpoint(corner_points+4, corner_points+6);
        Mat v_l = vline(vp1, vp2);
        
        Mat H_mat1 = (Mat_<double>(3,3)<< 1,0,0,0,1,0, \
            v_l.at<double>(0,0), v_l.at<double>(1,0), v_l.at<double>(2,0));
        
        //START step one correction
        //disp_mat(H_mat1);
        back_search_task2(fp2, img, H_mat1.inv());
        //END of step one
        
        // STEP 2 could be integrated HERE
        img.release();
        
    }
    // STEP 2    
    if (mouse_count >=8){
        setMouseCallback("testImage", NULL, NULL);
        
        Mat mat_S = calc_S(corner_points);
        //START step two correction
        Mat H_mat2 = build_affine_mat(mat_S);
        
        back_search_task2(fp2, img2, H_mat2);
        img.release();
        img2.release();
    }
    //SINGLE STEP METHOD

    if (mouse_count >=20){
        setMouseCallback("testImage", NULL, NULL);
        
        Mat mat_C = calc_H_one_step(corner_points);
        //START step two correction
        //Mat H_mat2 = build_affine_mat(mat_S);
        cout << mat_C << endl;
        back_search_task2(fp2, img, mat_C);
        img.release();
        img2.release();
    }
    
}
// connect points : BLACK LINE 
void drawLine( Mat img, Point start, Point end )
{
    int thickness = 3;
    int lineType = 8;
    line( img,
         start,
         end,
         Scalar( 0, 0, 0 ),
         thickness,
         lineType );
}

int main(int argc, const char * argv[])
{
    
    std::cout << "Hello, World!\n";
    /*read an image*/
    img = imread("Set5/Img3.jpg");
    img2 = imread("img3_step1.png");

    Point frame2_points[4];
    fp2 = frame2_points; 
    if (img.empty()) {
        cout<<"Error loading the image.." << endl;
    }
    namedWindow("testImage",1);
    /********* initialize **********/
    frame2_points[0].x = 0;
    frame2_points[0].y = 0;
    frame2_points[1].x = 0;
    frame2_points[1].y = img2.rows-1;//img.rows-1 --> only use img2 when doing STEP 2 
    frame2_points[2].x = img2.cols-1;//img.cols-1
    frame2_points[2].y = 0;
    frame2_points[3].x = img2.cols-1;//img.cols-1
    frame2_points[3].y = img2.rows-1;//img.rows-1
    cout << "finish loading..." <<endl;
    cout << "display H matrix.." << endl;
    
    //register mouse listener

    //Algorithm performed HERE
    setMouseCallback("testImage", callBackFunc, (void*)corner_points);
    // Select points UI
    imshow("testImage",  img);

    waitKey();
    

    return 0;
}

