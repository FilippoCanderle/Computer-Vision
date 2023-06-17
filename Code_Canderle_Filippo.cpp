//COMPUTER VISION A.A.2022-23, LABORATORY 4
//CANDERLE FILIPPO - MATRICOLA 2088236
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void pastePatch(const Mat& patch, const Mat& H, Mat& baseImage)
{
    Mat warpedPatch;
    warpPerspective(patch, warpedPatch, H, baseImage.size());

    Mat mask;
    cvtColor(warpedPatch, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 0, 255, THRESH_BINARY);

    warpedPatch.copyTo(baseImage, mask);
}
//From terminal run:
// cd /Users/filippo/Desktop/LAB4_CV; rm -rf build; mkdir build; cd build; cmake ..; make; ./MyProject
int main(int argc, char** argv){
    std::string dataset="international";
    Mat image = imread("/Users/filippo/Desktop/LAB4_CV/"+dataset+"/image_to_complete.jpg");
    cv::imshow("Image", image);
    cv::waitKey();

    int split=0;
    int k=4;
    string path = "/Users/filippo/Desktop/LAB4_CV/"+dataset+"/patch_";

    vector<vector<KeyPoint>> patchKeypoints;
    vector<Mat> patchDescriptors;
    vector<Mat> patches_arr;   
    for(int i = 0; i < k; i++) {

        string name = path + to_string(i) + ".jpg";
        Mat img = imread(name);
        string winName = "patch" + to_string(i);
        cv::imshow(winName, img);
        cv::waitKey();

        cv::Ptr<SURF> extractor = SURF::create();
        vector<KeyPoint> keypoints;
        Mat descriptors;

        extractor->detectAndCompute(img, Mat(), keypoints, descriptors);

        patchKeypoints.push_back(keypoints);
        patchDescriptors.push_back(descriptors);
        patches_arr.push_back(img);

        // Print the number of keypoints and descriptors
        cout << "Patch " << i << " - Keypoints: " << keypoints.size() << ", Descriptors: " << descriptors.rows << endl;
    }

    cv::Ptr<cv::xfeatures2d::SURF>extractor = SURF::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    extractor->detectAndCompute(image, Mat(), keypoints, descriptors);

    Mat img_keypoints;
    drawKeypoints(image, keypoints, img_keypoints);

    cv::imshow("Image - Keypoints", img_keypoints);
    cv::waitKey();
    cv::imwrite("/Users/filippo/Desktop/LAB4_CV/"+dataset+"/Keypoints.jpg", img_keypoints);

    
    cout << "Image - Keypoints: " << keypoints.size() << ", Descriptors: " << descriptors.rows << endl;

    cv::BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;

    vector<Point2f> basePoints;
    vector<Point2f> patchPoints;
    vector<DMatch> goodMatches;

    for (int i = 0; i < k; i++) {
        matcher.knnMatch(descriptors, patchDescriptors[i], matches, 2);


        const float ratioThreshold = 0.2f;
        vector<DMatch> goodMatches;
        for (size_t j = 0; j < matches.size(); j++) {
            if (matches[j][0].distance < ratioThreshold * matches[j][1].distance) {
                goodMatches.push_back(matches[j][0]);
            }
        }


        cout << "Matches - Patch " << i << ": " << goodMatches.size() << endl;

        for (const auto& match : goodMatches)
        {
            basePoints.push_back(keypoints[match.queryIdx].pt);
            patchPoints.push_back(patchKeypoints[i][match.trainIdx].pt);
        }
        
        
        if (split==1){
            Mat matchesDrawn;
            drawMatches(image, keypoints, patches_arr[i], patchKeypoints[i], goodMatches, matchesDrawn);

            namedWindow("Matches", cv::WINDOW_NORMAL);
            imshow("Matches", matchesDrawn);

            waitKey(0);
        }

        Mat H = findHomography(patchPoints, basePoints, RANSAC);

        pastePatch(patches_arr[i], H, image);

        // Pulizia dei vettori per la prossima patch
        basePoints.clear();
        patchPoints.clear();
        goodMatches.clear();

    }
    cv::imshow("Final Image", image);
    cv::imwrite("/Users/filippo/Desktop/LAB4_CV/"+dataset+"/final"+dataset+".jpg", image);
    cv::waitKey();   
}