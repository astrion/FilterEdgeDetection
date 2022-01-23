#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

int main()
{

    //https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/CV.jpg

    // Ignore logging messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);


//    Mat MyImage = Mat::zeros(100, 100, CV_8UC3);

    String ImageLocation = "../cv.jpg";  // provide message location string
    Mat ImgColor =  imread(ImageLocation, IMREAD_COLOR);  // read image color
//    Mat ImgGrayscale = imread(ImageLocation, IMREAD_GRAYSCALE);  // read image grayscale
//    Mat ImgUnchanged = imread(ImageLocation, IMREAD_UNCHANGED);  // read image unchanged

        // set auto image size
    namedWindow("color image", WINDOW_AUTOSIZE);
//    namedWindow("grayscale image", WINDOW_AUTOSIZE);
//    namedWindow("unchanged image", WINDOW_AUTOSIZE);

    imshow("color image", ImgColor);
//    imshow("grayscale image", ImgGrayscale);
//    imshow("unchanged image", ImgUnchanged);

//    imshow("mywindows", MyImage);


/*
The below part write a matrix into image
imwrite("grayscale.jpg", img_grayscale);
*/
    cv::waitKey(0);

    return 0;
}
