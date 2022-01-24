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
    //TODO: fix issue here
//    for (int y = 0; y < scaledRows; y++) {
//        for (int x = 0; x < scaledCols; x++) {
//            Vec3b &color = image.at<Vec3b>(y * scaleFactor, x * scaleFactor);
//
//            //set pixel
//            scaledImg.at<Vec3b>(Point(x, y)) = color;
//        }
//    }
    return scaledImg;
}

Mat imageShifter(Mat image, int numShiftPixels) {
    /* shift image to the top right */

    // perform a scaling of rows and cols

    Mat shiftedImg = Mat::zeros(Size(image.cols, image.rows), CV_8UC3);

    // iterate through 2D matrices
    for (int y = 0; y < image.cols-numShiftPixels; y++) {
        for (int x = 0; x < image.rows-numShiftPixels; x++) {
            Vec3b &color = image.at<Vec3b>(y, x);

            //set pixel
            shiftedImg.at<Vec3b>(Point(x, y)) = color;
        }
    }
    return shiftedImg;
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
    namedWindow("p1-c: upscaled image with linear", WINDOW_AUTOSIZE);
    namedWindow("p1-c: upscaled image with cubic", WINDOW_AUTOSIZE);

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
    Mat shiftedImg = imageShifter(img1, 30);
    namedWindow("p2-a: shifted image to top right", WINDOW_AUTOSIZE);
    imshow("p2-a: shifted image to top right", shiftedImg);
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



