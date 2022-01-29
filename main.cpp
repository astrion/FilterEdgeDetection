#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat manualDownScale(Mat image, int scaleFactor) {
    // perform a scaling of rows and cols
    int scaledRows = image.rows / scaleFactor;
    int scaledCols = image.cols / scaleFactor;

    Mat scaledImg = Mat::zeros(Size(scaledCols, scaledRows), CV_8UC3);

    // iterate through 2D matrices
    for (int y = 0; y < scaledRows; y++) {
        for (int x = 0; x < scaledCols; x++) {
            Vec3b &color = image.at<Vec3b>(y * scaleFactor, x * scaleFactor);

            //set pixel
            scaledImg.at<Vec3b>(Point(x, y)) = color;
        }
    }
    return scaledImg;
}

Mat imageShifterXY(Mat image, int shiftX, int shiftY) {
    /* shift image to the top right */
    //Move the image along x and y axis through regular quadrant system.

    Mat shiftedImg = Mat::zeros(Size(image.cols, image.rows), CV_8UC3);
    int xStart = 0;
    int yStart = 0;
    int xEnd = image.cols;
    int yEnd = image.rows;

    if (shiftY > 0)
         yEnd -= shiftY;
    else if(shiftY <0)
        yStart -= shiftY;

    if (shiftX > 0)
        xStart += shiftX;
    else if(shiftX <0)
        xEnd += shiftX;

        // iterate through 2D matrices
    for (int y = yStart; y < yEnd; y++) {
        for (int x = xStart; x < xEnd; x++) {
            Vec3b &color = image.at<Vec3b>(y, x);

            //set pixel
            shiftedImg.at<Vec3b>(Point(x, y)) = color;
        }
    }
    return shiftedImg;
}

Mat padder(Mat image, int pad){
    //pad around the image by the given 'pad' amount
    Mat paddedImg = Mat::zeros(Size(image.cols+pad*2, image.rows+pad*2), CV_8UC3);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b &color = image.at<Vec3b>(y, x);
            paddedImg.at<Vec3b>(Point(x+pad, y+pad)) = color;
        }
    }
    return paddedImg;
}

Mat gaussianKernel(int dim){
    Mat A =Mat::zeros(Size(dim, dim),CV_32FC1);
    A.at<float>(0,0)=0.11;
    A.at<float>(0,1)=0.11;
    A.at<float>(0,2)=0.11;
    A.at<float>(1,0)=0.11;
    A.at<float>(1,1)=0.11;
    A.at<float>(1,2)=0.11;

    return A;
}

Mat gaussianFilter(Mat image) {
    int dim = 3;

    //pad image
    Mat padded = padder(image, dim / 2);

   //generate kernel
    Mat kernel = gaussianKernel(3);

    //apply kernel
    Mat applied = Mat::zeros(Size(image.cols, image.rows), CV_8UC3);

    //Mat::convertTo(newImage, CV_32FC3, 1/255.0);
//cv::Matx33f
//https://titanwolf.org/Network/Articles/Article?AID=fe4e203f-9cb1-40a7-b39d-9e55f387be87
//https://riptutorial.com/opencv/example/9922/efficient-pixel-access-using-cv--mat--ptr-t--pointer
//    ushort * ptr = applied.ptr<ushort>();
    Mat t =Mat::zeros(Size(dim, dim),CV_8UC3);
    for (int y=0; y<image.rows; y++){
        Vec3b *p = padded.ptr<cv::Vec3b>(y);
        Vec3b *k = applied.ptr<cv::Vec3b>(y);
        for (int x=0; x<image.cols; x++){
//        ptr[9*i+0] = image.data[i+0];

            // Matrix multiplication
//            for(int j=0; j<dim; j++){
//                for(int k=0; j<dim; k++){

//                vector<Vec3b> &color = padded.data.at<Vec3b>(y, x);
//                ptr[x] = cv::Vec3b(ptr[x][2], ptr[x][1], ptr[x][0]);

//            ptrA[x] = cv::Vec3b(ptr[x][2], ptr[x][1], ptr[x][0]);
            k[x] = cv::Vec3b(p[x]);

//            applied.at<Vec3b>(Point(x, y)) = color;
//            }

            //apply kernel



        }
    }
    return applied;
}

void runPart1() {
    //https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/CV.jpg

    String ImageLocation1 = "../cv.jpg ";  // provide message location string
    String ImageLocation2 = "../rgb.png ";  // provide message location string
    Mat img1 = imread(ImageLocation1, IMREAD_COLOR);  // read image color
    Mat img2 = imread(ImageLocation2, IMREAD_COLOR);  // read image color


    Mat downScaledImg2 = manualDownScale(img1, 2);  // perform a scaling operation
    Mat downScaledImg4 = manualDownScale(img1, 4);  // perform a scaling operation
    Mat downScaledImg8 = manualDownScale(img1, 8);  // perform a scaling operation
    Mat downScaledImg16 = manualDownScale(img1, 16);  // perform a scaling operation

    // set auto image size
    namedWindow("p1-a,b: 1/2 scaled image", WINDOW_AUTOSIZE);
    namedWindow("p1-a,b: 1/4 scaled image", WINDOW_AUTOSIZE);
    namedWindow("p1-a,b: 1/8 scaled image", WINDOW_AUTOSIZE);
    namedWindow("p1-a,b: 1/16 scaled image", WINDOW_AUTOSIZE);

    imshow("p1-a,b: 1/2 scaled image", downScaledImg2);
    imshow("p1-a,b: 1/4 scaled image", downScaledImg4);
    imshow("p1-a,b: 1/8 scaled image", downScaledImg8);
    imshow("p1-a,b: 1/16 scaled image", downScaledImg16);


    int upScaleFactor = 10;
    Mat upScaledImgNearest;
    Mat upScaledImgLinear;
    Mat upScaledImgCubic;


    resize(img2,
           upScaledImgNearest,
           Size(img2.rows * upScaleFactor, img2.cols * upScaleFactor),
           INTER_NEAREST);
    resize(img2,
           upScaledImgLinear,
           Size(img2.rows * upScaleFactor, img2.cols * upScaleFactor),
           INTER_LINEAR);
    resize(img2,
           upScaledImgCubic,
           Size(img2.rows * upScaleFactor, img2.cols * upScaleFactor),
           INTER_CUBIC);

    // set auto image size
    namedWindow("p1-c: upscaled image with nearest interpolation", WINDOW_AUTOSIZE);
    namedWindow("p1-c: upscaled image with linear interpolation", WINDOW_AUTOSIZE);
    namedWindow("p1-c: upscaled image with cubic interpolation", WINDOW_AUTOSIZE);

    imshow("p1-c: upscaled image with nearest interpolation", upScaledImgNearest);
    imshow("p1-c: upscaled image with linear interpolation", upScaledImgLinear);
    imshow("p1-c: upscaled image with cubic interpolation", upScaledImgCubic);

//    imshow("mywindows", MyImage);



/*
The below part write a matrix into image
imwrite("grayscale.jpg", img_grayscale);
*/
}

void runPart2() {
    String ImageLocation1 = "../cv.jpg ";  // provide message location string
    Mat img1 = imread(ImageLocation1, IMREAD_COLOR);  // read image color
    Mat shiftedImg = imageShifterXY(img1, 50, 50);
    namedWindow("p2-a: shifted image to top right", WINDOW_AUTOSIZE);
    imshow("p2-a: shifted image to top right", shiftedImg);

    Mat img2 = imread(ImageLocation1, IMREAD_COLOR);  // read image color

    Mat newImg = gaussianFilter(img2);
    namedWindow("p2-b: gaussian filtered image", WINDOW_AUTOSIZE);
    imshow("p2-b: gaussian filtered image", newImg);
}

void runPart3() {

}

void runPart4() {

}

void runPart5() {

}

int main(int argc, char **argv) {
    // Ignore logging messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    switch (std::stoi(argv[1])) {
        case 1:
            runPart1();
            break;
        case 2:
            runPart2();
            break;
        case 3:
            runPart3();
            break;
        case 4:
            runPart4();
            break;
        case 5:
            runPart5();
            break;
        default:
            break;
    }
    cv::waitKey(0);
    return 0;
}



