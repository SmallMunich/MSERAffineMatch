#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <vector>
#include "vfc.h"

using namespace std;
using namespace cv;

// 在彩色图像上绘制检测到特征点椭圆
void draw_ellipse(Mat& image, vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, Mat& image_rgb);
// Elliptic_KeyPoint 转换至标准 KeyPoint
bool ConvertEllipseKeyPointToKeyPoint(const vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, vector<KeyPoint>& kpts);

int main(void)
{
	//-- 以灰度模式读取图像
	Mat img1 = imread("../opencv_affine/image/adam_zoom1_front.png", 0);
	Mat img2 = imread("../opencv_affine/image/adam_zoom1_45deg.png", 0);
	
	Ptr<MSER> feature2D = MSER::create(1, 100, 10000, 0.25);  // MSER特征点检测算子

	//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::SIFT::create();  // SIFT 特征描述子
	Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::DAISY::create();  // DAISY描述子

	//-- Step 1: 计算特征点
	vector<xfeatures2d::Elliptic_KeyPoint> aff_kpts1, aff_kpts2;
	// 将定义的特征点提取方式与描述符构建方式初始化AffineFeature2D
	Ptr<xfeatures2d::AffineFeature2D> affineFeature2D = xfeatures2d::AffineFeature2D::create(feature2D, descriptor2D);
	//-- 可以先提取特征点信息
	//affineFeature2D->detect(img1, aff_kpts1);
	//affineFeature2D->detect(img2, aff_kpts2);

	//-- Step 2: 计算特征点与描述符信息
	Mat desc1, desc2;
	//-- 如果事先提取好Elliptic_KeyPoint特征点信息，那么将detectAndCompute()函数最后一个入参改为true
	affineFeature2D->detectAndCompute(img1, Mat(), aff_kpts1, desc1, false);
	affineFeature2D->detectAndCompute(img2, Mat(), aff_kpts2, desc2, false);
	//-- 数据结构转换
	vector<KeyPoint> kpts1, kpts2;
	ConvertEllipseKeyPointToKeyPoint(aff_kpts1, kpts1);
	ConvertEllipseKeyPointToKeyPoint(aff_kpts2, kpts2);

	//-- Step 3: 通过描述符来匹配特征点一致性
	FlannBasedMatcher matcher;
	vector<DMatch> matches;

	matcher.match(desc1, desc2, matches);

	//-- Step 4: 匹配点对精确提纯 (VFC)
	vector<Point2f> X;  vector<Point2f> Y;
	X.clear();          Y.clear();
	for (unsigned int i = 0; i < matches.size(); i++) {
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		X.push_back(kpts1[idx1].pt);
		Y.push_back(kpts2[idx2].pt);
	}
	// VFC process
	double t = (double)getTickCount();
	VFC myvfc;
	myvfc.setData(X, Y);
	myvfc.optimize();
	vector<int> matchIdx = myvfc.obtainCorrectMatch();
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "VFC Times (s): " << t << endl;

	vector< DMatch > correctMatches;
	correctMatches.clear();
	for (unsigned int i = 0; i < matchIdx.size(); i++) {
		int idx = matchIdx[i];
		correctMatches.push_back(matches[idx]);

	}

	// 获取经过精细匹配后的特征点椭圆相关参数信息
	vector<xfeatures2d::Elliptic_KeyPoint> inliers_kpts1, inliers_kpts2;
	for (unsigned int i = 0; i < correctMatches.size(); ++i)
	{
		int idx1 = correctMatches[i].queryIdx;
		int idx2 = correctMatches[i].trainIdx;
		inliers_kpts1.push_back(aff_kpts1[idx1]);
		inliers_kpts2.push_back(aff_kpts2[idx2]);
	}
	// 绘制精确匹配点对局部椭圆区域
	Mat img1_rgb, img2_rgb;
	draw_ellipse(img1, inliers_kpts1, img1_rgb);
	draw_ellipse(img2, inliers_kpts2, img2_rgb);

	Mat img_correctMatches;
	drawMatches(img1_rgb, kpts1, img2_rgb, kpts2, correctMatches, img_correctMatches, Scalar::all(-1), \
		        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("PreciseMatchWithVFC");
	imshow("PreciseMatchWithVFC", img_correctMatches);
	imwrite("../opencv_affine/image/match_img15.png", img_correctMatches);
	waitKey(0);

	return 0;
}

void draw_ellipse(Mat& image, vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, Mat& image_rgb)
{
	if (1 == image.channels()) {
		image_rgb = Mat(Size(image.cols, image.rows), CV_8UC3);
		cvtColor(image, image_rgb, COLOR_GRAY2BGR);
	}
	else {
		image.copyTo(image_rgb);
	}
	Point center;
	Size axes;

	for (int i = 0; i<elliptic_keypoints.size(); ++i)
	{
		center.x = elliptic_keypoints[i].pt.x;
		center.y = elliptic_keypoints[i].pt.y;
		axes.width = elliptic_keypoints[i].axes.width;
		axes.height = elliptic_keypoints[i].axes.height;
		double angle = elliptic_keypoints[i].angle;
		// 绘制椭圆图像
		ellipse(image_rgb, center, axes, angle * 180 / CV_PI, 0, 360, Scalar(0, 255, 0), 1, 8);
		// 绘制中心点坐标
		circle(image_rgb, center, 1, Scalar(0, 0, 255));
	}
}

bool ConvertEllipseKeyPointToKeyPoint(const vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, vector<KeyPoint>& kpts)
{
	if (0 == elliptic_keypoints.size())
		return false;
	for (int i = 0; i < elliptic_keypoints.size(); ++i)
	{
		KeyPoint kpt;
		kpt.pt.x = elliptic_keypoints[i].pt.x;
		kpt.pt.y = elliptic_keypoints[i].pt.y;
		kpt.angle = elliptic_keypoints[i].angle;
		float diam = elliptic_keypoints[i].axes.height*elliptic_keypoints[i].axes.width;
		kpt.size = sqrt(diam);
		kpts.push_back(kpt);
	}
	return true;
}

//Mat showMatch;
//drawMatches(img1, kpts1, img2, kpts2, matches, showMatch, Scalar::all(-1), \
	//	        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//namedWindow("RoughMatchWithOutVFC");
//imshow("RoughMatchWithOutVFC", showMatch);