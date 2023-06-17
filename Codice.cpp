
//LAB2 COMPUTER VISION A.Y.22-23
//Canderle Filippo, 2088236

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

/*From Command line run the following:

cd /Users/filippo/Desktop/Lab2_CV; rm -rf build; mkdir build; cd build; cmake ..; make; ./MyProject

And everything should work correctly.
*/

void showHistogram(std::vector<cv::Mat>& hists)
{
  // Min/Max computation
  double h_max[3] = {0,0,0};
  double min;
  cv::minMaxLoc(hists[0], &min, &h_max[0]);
  cv::minMaxLoc(hists[1], &min, &h_max[1]);
  cv::minMaxLoc(hists[2], &min, &h_max[2]);

  std::string wname[3] = { "Blue", "Green", "Red" };
  cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255) };

  std::vector<cv::Mat> canvas(hists.size());

  // Display each histogram in a canvas
  for (int i = 0, end = hists.size(); i < end; i++)
  {
    canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
    {
      cv::line(
            canvas[i],
            cv::Point(j, rows),
            cv::Point(j, rows - (hists[i].at<float>(j) * rows/h_max[i])),
            hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
            1, 8, 0
            );
    }

    cv::imshow(hists.size() == 1 ? "Value" : wname[i], canvas[i]);
    cv::waitKey();
  }
}


int main(int argc, char** argv){


    //Load an image and show it.
    Mat image = imread("/Users/filippo/Desktop/Lab2_CV/data/dei.jpg");
    cv::imshow("Image", image);
    cv::waitKey();
    
    //Before computing the histogram of R, G, B channels , I define the parameters
    const int histSize = 256;
    vector<Mat> bgr_planes;
    split(image, bgr_planes);
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    //I actually compute the histograms...
    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    //...and call the provided function showHistogram()
    std::vector<cv::Mat> tot_hist = { b_hist, g_hist, r_hist };
    showHistogram(tot_hist);

    //Now I equalize the r,g,b components using equalizeHist(). Firstly I split the three components of the image.
    Mat bgr[3], equalized[3];
    split(image, bgr);

    // Equalize each BGR component separately
    for (int i = 0; i < 3; i++) {
        equalizeHist(bgr[i], equalized[i]);
    }

    // Merge equalized BGR components back into original image
    Mat equalized_image;
    merge(equalized, 3, equalized_image);

    imshow("Equalized Image", equalized_image);
    cv::waitKey();
    imwrite("/Users/filippo/Desktop/Lab2_CV/data/filtering/equalized.jpg", equalized_image);
    //...But, as we can notice there are some regions with artifacts.

    //So I convert the image in a new color space and I perform the equalization of luminance channel
    cv::Mat img_nc = cv::imread("/Users/filippo/Desktop/Lab2_CV/data/dei.jpg");
    cv::Mat img_lab;
    cv::cvtColor(img_nc, img_lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(img_lab, lab_channels);

    cv::Mat img_lab_l_eq;
    cv::equalizeHist(lab_channels[0], img_lab_l_eq);

    cv::Mat out_image_lab;
    std::vector<cv::Mat>img_l_eq_vec = {img_lab_l_eq, lab_channels[1], lab_channels[2]};
    cv::merge(img_l_eq_vec, out_image_lab);

    //Shows the equalized image
    cv::Mat out_image_eql_rgb;
    cv::cvtColor(out_image_lab, out_image_eql_rgb, cv::COLOR_Lab2BGR);
    cv::imshow("Equalized image in Lab space", out_image_eql_rgb);
    cv::waitKey();
    imwrite("/Users/filippo/Desktop/Lab2_CV/data/filtering/equalized_labspace.jpg", out_image_eql_rgb);

    //Apply the filters
    //Firstly I apply bilateral filter, Asking the required 3 parameters in input
    cout<<"APPLY BILATERAL FILTER"<<endl;
    cout<<"Enter sizeBR(double)"<<endl;
    double sizeBR;
    cin>>sizeBR;
    cout<<"Enter kernelBR(double)"<<endl;
    double kernelBR;
    cin>>kernelBR;
    cout<<"Enter space(double)"<<endl;
    double space;
    cin>>space;
    //I check the parameters and eventually correct them, in order to avoid possible errors.
    if ((int)(kernelBR) % 2 == 0)
        kernelBR = kernelBR - 1;
    if ((int)(space) % 2 == 0)
        space = space - 1;

    //I actually apply the filter to the image.
    Mat img_bil_range = image.clone();
    Mat bilater_range_filter_img;
    bilateralFilter(img_bil_range, bilater_range_filter_img, sizeBR, kernelBR, space);
    imshow("Range bilateral", bilater_range_filter_img);
    cv::waitKey();
    //Save the filtered image
    imwrite("/Users/filippo/Desktop/Lab2_CV/data/filtering/range_bilateral.jpg", bilater_range_filter_img);

    //After I apply median filter on a brand new image, asking the parameter in input
    cout<<"APPLY MEDIAN FILTER"<<endl;
    cout<<"Enter kernelM(int)"<<endl;
    int kernelM;
    cin>>kernelM;
    if ((int)(kernelM) % 2 == 0)
        kernelM = kernelM - 1;
    //I actually apply the filter to the image.
    Mat img_median = image.clone();
    Mat median_filter;
    cv::medianBlur(img_median, median_filter, kernelM);
    imshow("median", median_filter);
    cv::waitKey();
    //Save the filtered image
    imwrite("/Users/filippo/Desktop/Lab2_CV/data/filtering/median_filter.jpg", median_filter);
    
    //Finally I apply gaussian filter on a brand new image, asking 2 parameters in input.
    cout<<"APPLY GAUSSIAN FILTER"<<endl;
    cout<<"Enter wdt(int)"<<endl;
    int wdt;
    cin>>wdt;
    cout<<"Enter hgt(int)"<<endl;
    int hgt;
    cin>>hgt;
    if (wdt % 2 == 0)
        wdt = wdt - 1;
    if (hgt % 2 == 0)
        hgt = hgt -1;
    Size sizeGB(wdt, hgt);

    //I actually apply the filter to the image.
    Mat img_gauss_kern = image.clone();
    Mat gauss_kern;
    GaussianBlur(img_gauss_kern, gauss_kern, sizeGB, 0);
    imshow("kernel", gauss_kern);
    cv::waitKey();

    //Save the filtered image.
    imwrite("/Users/filippo/Desktop/Lab2_CV/data/filtering/kernel_filter.jpg", gauss_kern);
    
    //I close all the opened windows.
    destroyAllWindows();
}