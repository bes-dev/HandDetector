#include "HandDetector.h"

using namespace cv;
using namespace std;

/*******************************************************
********************************************************
*********************HandDetector***********************
********************************************************
*******************************************************/

void HandDetector::setParams(HandDetector::Params& p)
{
	param.area = p.area;
	param.cosThreshold = p.cosThreshold;
	param.equalThreshold = p.equalThreshold;
	param.r = p.r;
	param.step = p.step;
}

bool HandDetector::isEqual(double a, double b)
{
	return fabs(a - b) <= param.equalThreshold;
}

double HandDetector::angle(std::vector<cv::Point>& contour, int pt, int r)
{
	int size = contour.size();
	cv::Point p0=(pt>0)?contour[pt%size]:contour[size-1+pt];
	cv::Point p1=contour[(pt+r)%size];
	cv::Point p2=(pt>r)?contour[pt-r]:contour[size-1-r];

	double ux=p0.x-p1.x;
	double uy=p0.y-p1.y;
	double vx=p0.x-p2.x;
	double vy=p0.y-p2.y;
	return (ux*vx + uy*vy)/sqrt((ux*ux + uy*uy)*(vx*vx + vy*vy));
}

signed int HandDetector::rotation(std::vector<cv::Point>& contour, int pt, int r)
{
	int size = contour.size();
	cv::Point p0=(pt>0)?contour[pt%size]:contour[size-1+pt];
	cv::Point p1=contour[(pt+r)%size];
	cv::Point p2=(pt>r)?contour[pt-r]:contour[size-1-r];

	double ux=p0.x-p1.x;
	double uy=p0.y-p1.y;
	double vx=p0.x-p2.x;
	double vy=p0.y-p2.y;
	return (ux*vy - vx*uy);
}

void HandDetector::detect(cv::Mat& mask, std::vector<Hand>& hands)
{
	hands.clear();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(mask.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	if(!contours.empty())
	{
		for(int i=0; i<contours.size(); i++)
		{
			if(cv::contourArea(contours[i])>param.area)
			{
				Hand tmp;
				cv::Moments m=cv::moments(contours[i]);
				tmp.center.x=m.m10/m.m00;
				tmp.center.y=m.m01/m.m00;

				for(int j = 0; j < contours[i].size(); j+= param.step )
				{
					double cos0 = angle (contours[i], j, param.r);

					if ((cos0 > 0.5)&&(j+param.step<contours[i].size()))
					{
						double cos1 = angle (contours[i], j - param.step, param.r);
						double cos2 = angle (contours[i], j + param.step, param.r);
						double maxCos = std::max(std::max(cos0, cos1), cos2);
						bool equal = isEqual (maxCos , cos0);
						signed int z = rotation (contours[i], j, param.r);
						if (equal == 1 && z<0)
						{
							tmp.fingers.push_back(contours[i][j]);
						}
					}
				}
				tmp.contour=contours[i];
				hands.push_back(tmp);
			}
		}
	}
}


void drawHands(cv::Mat& image, std::vector<Hand>& hands)
{
	int size = hands.size();
	std::vector<std::vector<cv::Point>> c;
	for(int i = 0; i<size; i++)
	{
		c.clear();
		c.push_back(hands[i].contour);
		cv::circle(image, hands[i].center, 20, cv::Scalar(0, 0, 255), 2);
		int fingersSize = hands[i].fingers.size();
		for(int j = 0; j < fingersSize; j++)
		{
			cv::circle(image, hands[i].fingers[j], 10, cv::Scalar(0, 0, 255), 2);
			cv::line(image, hands[i].center, hands[i].fingers[j], cv::Scalar(0, 0, 255), 4);
		}
		std::cout<<hands[i].fingers.size()<<endl;
	}
}