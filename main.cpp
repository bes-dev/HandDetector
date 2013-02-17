#include <iostream>
#include "HandDetector.h"

using namespace std;
using namespace cv;

int main()
{
	cv::VideoCapture cap(CV_CAP_OPENNI);
	cv::namedWindow("depth", 1);
	cv::namedWindow("bgr", 1);
	
	HandDetector::Params p;
	p.area=1000;
	p.cosThreshold=0.5;
	p.equalThreshold=1e-7;
	p.r=40;
	p.step=16;

	HandDetector hDetector;
	hDetector.setParams(p);
	std::vector<Hand> hands;

	while(1)
	{
		cv::Mat depthMap;
		cv::Mat bgrImage;

		cap.grab();

		cap.retrieve( depthMap, CV_16UC1 );
		cap.retrieve( bgrImage, CV_32FC1 );

		cv::Mat tmp;
		cv::cvtColor(depthMap, tmp, CV_GRAY2BGR);

		cv::threshold(depthMap, depthMap, 60, 255, cv::THRESH_BINARY);

		hDetector.detect(depthMap, hands);

		if(!hands.empty())
		{
			drawHands(tmp, hands);
			drawHands(bgrImage, hands);
		}

		cv::imshow("depth", tmp);
		cv::imshow("bgr", bgrImage);

		if( cv::waitKey( 20 ) >= 0 )
			break;
	}

	cv::waitKey();
	return 0;
}