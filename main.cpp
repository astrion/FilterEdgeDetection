#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

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
    else if (shiftY < 0)
        yStart -= shiftY;

    if (shiftX > 0)
        xStart += shiftX;
    else if (shiftX < 0)
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


Mat padder(Mat image, int pad) {
    //pad around the image by the given 'pad' amount
    Mat paddedImg;
    paddedImg.create(image.rows + 2 * pad, image.cols + 2 * pad, image.type());
    paddedImg.setTo(Vec3f::all(0));
    image.copyTo(paddedImg(Rect(pad, pad, image.cols, image.rows)));
    return paddedImg;
}


Mat gaussianKernel(int dim) {
    Mat A = Mat::zeros(Size(dim, dim), CV_32FC3);
    A.at<Vec3f>(0, 0) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(0, 1) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(0, 2) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(1, 0) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(1, 1) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(1, 2) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(2, 0) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(2, 1) = Vec3f(0.11, 0.11, 0.11);
    A.at<Vec3f>(2, 2) = Vec3f(0.11, 0.11, 0.11);
    return A;
}

Mat multiplication(Mat &A, Mat &B) {
    return A * B;
}

Mat gaussianFilter(Mat image) {
    image.convertTo(image, CV_32FC3, 1 / 255.0);

    int dim = 3;

    //pad image
    Mat padded = padder(image, dim / 2);

    //generate kernel
    Mat kernel = gaussianKernel(3);

    //apply kernel
    Mat applied(Size(image.cols, image.rows), CV_32FC3, Scalar::all(0.f));

//cv::Matx33f
//https://titanwolf.org/Network/Articles/Article?AID=fe4e203f-9cb1-40a7-b39d-9e55f387be87
//https://riptutorial.com/opencv/example/9922/efficient-pixel-access-using-cv--mat--ptr-t--pointer
//    ushort * ptr = applied.ptr<ushort>();
    Mat t = Mat::zeros(Size(dim, dim), CV_8UC3);
    int halfDim = dim / 2;
    for (int y = 0; y < image.rows; y++) {
//        Vec3b *p = padded.ptr<cv::Vec3b>(y);
//        Vec3b *k = applied.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; x++) {


//             Matrix multiplication
            Mat small(dim, dim, CV_32FC3, Scalar::all(0.f));
            for (int j = -halfDim; j < halfDim; j++) {
                for (int k = -halfDim; k < halfDim; k++) {
                    int _y = y + j + halfDim;
                    int _x = x + k + halfDim;
                    Vec3f &bgr = padded.at<Vec3f>(_y, _x);
                    small.at<Vec3f>(Point(j + halfDim, k + halfDim)) = bgr;
                }
            }
            Mat tmp(dim, dim, CV_32FC3, Scalar::all(0.f));

            cv::multiply(small, kernel, tmp);
            //TODO: issue with three channel multiplications.
            // this link could be helpful: https://answers.opencv.org/question/66216/how-to-multiply-cvmat-with-mask/
//            applied.at<Vec3f>(Point(y, x)) = tmp;
//                    k[x] = cv::Vec3b(p[x]);
//            applied.at<Vec3b>(Point(x, y)) = color;
//            }
            //apply kernel
        }
    }

    padded.convertTo(padded, CV_8UC3, 255);
    return padded;
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



