#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

int is_oof(Mat img){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type())+255;

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    int check = 0;
    //-- Keep only those contours that have a point inside roi
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test == 0){
      			check = 1;
      		}
       	}
    }
	return check;
}

vector<Point> keep_roi(Mat img,Point tl, Point br, Mat &mask){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type());
    rectangle(src,tl,br,255,CV_FILLED);

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Keep only those contours that have a point inside roi
    vector<Point> cc;
    Mat kept = Mat::zeros(img.size(),img.type());
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test==1 || test == 0){
      			for(unsigned int k=0; k<contours[i].size(); k++){
      				cc.push_back(contours[i][k]);
      			}
      			drawContours(kept, contours, i, 255, CV_FILLED);
      			break;
      		}
       	}
    }
	Mat kept_mask;
	bitwise_and(img,kept,kept_mask);

    mask = kept_mask;
	return cc;
}

float get_fd(Mat mask){
	//-- Need to remap the image to 2048x2048 so box counting can be used
	Mat img_bc;
	resize(mask, img_bc, Size(2048,2048), 0, 0, INTER_LINEAR);

	//-- Initializing variables
	double width = 2048.0;
	double p = log(width)/log(double(2.0));
	VectorXf N = VectorXf::Zero(int(p)+1);
	double sumImg = sum(img_bc)[0];
	N(int(p)) = sumImg;

	//-- Boxcounting
	double siz;
	double siz2;
	float running_sum;
    for (int g = int(p)-1; g > 0; g--){
    	siz = pow(2.0, double(p-g));
    	siz2 = round(siz/2.0);
    	running_sum = 0;
    	for (int i = 0; i < int(width-siz+1); i = i+int(siz)){
    		for (int j = 0; j < int(width-siz+1); j = j+int(siz)){
    			img_bc.at<uchar>(i,j) = (bool(img_bc.at<uchar>(i,j)) || bool(img_bc.at<uchar>(i+siz2,j))
    				|| bool(img_bc.at<uchar>(i,j+siz2)) || bool(img_bc.at<uchar>(i+siz2,j+siz2)));
    			running_sum = running_sum+float(img_bc.at<uchar>(i,j));
    		}
    	}
    	N(g) = running_sum;
	}
    N = N.colwise().reverse().eval();

    //-- Getting bin sizes
    VectorXf R = VectorXf::Zero(int(p)+1);
    R(0) = 1.0;
    for (int k = 1; k < R.size(); k++){
    	R(k) = pow(2.0, double(k));
    }

    //-- Calculating log-log slopes
	float slope [R.size()-1];
	for(int i=1;i < R.size()-1 ;i++){
		slope[i] = (log10(N(i+1))-log10(N(i)))/(log10(R(i+1))-log10(R(i)));
	}

	//-- Getting average slope (fractal dimension)
	float sum = 0.0, average;
	int s_count =0;
	for(int i=1; i < R.size()-1; i++){
		if(-slope[i] < 2 && -slope[i] > 0){
			sum += -slope[i];
			s_count++;
		}
	}
	average = sum / s_count;
	return average;
}

vector<double> get_shapes(vector<Point> cc,Mat mask){
    //-- Get measurements
    Moments mom = moments(mask,true);
    double area = mom.m00;
    vector<Point>hull;
    convexHull( Mat(cc), hull, false );
    double hull_verticies = hull.size();
    double hull_area = contourArea(Mat(hull));
    double solidity = area/hull_area;
    double perimeter = arcLength(Mat(cc),false);
    double cmx = mom.m10 / mom.m00;
    double cmy = mom.m01 / mom.m00;
    Rect boundRect = boundingRect( cc );
    double width = boundRect.width;
    double height = boundRect.height;
    double circ = 4*M_PI*area/(perimeter*perimeter);
    double angle = -1;
    double ex = -1;
    double ey = -1;
    double emajor = -1;
    double eminor = -1;
    double eccen = -1;
    double round = -1;
    double ar = -1;
    if(cc.size() >= 6){
        Mat pointsf;
    	Mat(cc).convertTo(pointsf, CV_32F);
   	    RotatedRect ellipse = fitEllipse(pointsf);
   	    angle = ellipse.angle;
  	    ex = ellipse.center.x;
   	    ey = ellipse.center.y;
   	    if(ellipse.size.height > ellipse.size.width){
   	    	emajor = ellipse.size.height;
   	    	eminor = ellipse.size.width;
   	    }else{
   	    	eminor = ellipse.size.height;
  	   	    emajor = ellipse.size.width;
   	    }
   	    eccen = sqrt((1- eminor / emajor)*2);
   	    round = eminor/emajor;
   	    ar = emajor/eminor;
    }
    float fd = get_fd(mask);
    double oof = is_oof(mask);
    double shapes[20] = {area,hull_area,solidity,perimeter,width,height,cmx,cmy,hull_verticies,ex,ey,emajor,eminor,angle,eccen,circ,round,ar,fd,oof};
    vector<double> shapes_v(shapes,shapes+20);
    return shapes_v;
}

Mat get_color(Mat img,Mat mask){
	Mat composite;
	cvtColor(img,composite,COLOR_BGR2HSV);
    vector<Mat> channels1;
    split(composite, channels1);
    Mat hist;
	int dims = 1; // Only 1 channel, the hue channel
	int histSize = 180; // 180 bins, actual range is 0-360.
	float hranges[] = { 0, 180 }; // hue varies from 0 to 179, see cvtColor
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&channels1[0],1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

void split(const string& s, char c, vector<string>& v) {
   string::size_type i = 0;
   string::size_type j = s.find(c);

   while (j != string::npos) {
      v.push_back(s.substr(i, j-i));
      i = ++j;
      j = s.find(c, j);

      if (j == string::npos)
         v.push_back(s.substr(i, s.length()));
   }
}

Mat get_gray(Mat img,Mat mask){
    Mat hist;
	int dims = 1;
	int histSize = 255;
	float hranges[] = { 0, 255 };
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&img,1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

int main(int argc, char** argv){
	string mode;
	if(argc == 1){
		mode = "-h";
	}else{
		mode = string(argv[1]);
	}
	bool bool_vis = mode=="VIS";
	bool bool_visD = mode=="VIS_DEBUG";
	bool bool_avgImgs = mode=="AVG_IMGS";
	bool bool_hyper = mode=="HYPER";
	bool bool_hyperD = mode=="HYPER_DEBUG";

	if(bool_vis || bool_visD){
		if(argc != 5){
			cout << "Using mode VIS requires input in this order: inputImage shapes_file.txt color_file.txt" << endl;
		}
		else{
			Mat inputImage = imread(argv[2]);
			Mat adjImage;
	    	cvtColor(inputImage, adjImage, CV_BGRA2BGR);

	    	//-- Thresholding b from Lab
			Mat lab;
			cvtColor(adjImage, lab, CV_BGR2Lab);
	    	vector<Mat> split_lab;
	    	split(lab, split_lab);
	    	Mat b_thresh;
	    	inRange(split_lab[2],0,143,b_thresh);
	    	Mat mask_b =  cv::Scalar::all(255) - b_thresh;

	    	//-- Thresholding s from HSV
			Mat hsv;
			cvtColor(adjImage, hsv, CV_BGR2HSV);
	    	vector<Mat> split_hsv;
	    	split(hsv, split_hsv);
	    	Mat s_thresh;
	    	inRange(split_hsv[1],0,65,s_thresh);
	    	Mat mask_s =  cv::Scalar::all(255) - s_thresh;

	    	//-- joining mask_b with mask_s and closing
	    	Mat mask_and;
	    	bitwise_and(mask_b,mask_s,mask_and);
	    	Mat mask_dilate;
	    	dilate(mask_and, mask_dilate, Mat(), Point(-1, -1), 3, 1, 1);
	    	Mat mask_erode;
	    	erode(mask_dilate,mask_erode, Mat(), Point(-1, -1), 3, 1, 1);

		    //-- ROI selector
	    	Mat mask;
	    	vector<Point> cc = keep_roi(mask_erode,Point(507,270),Point(2181,1731),mask);

	    	if(bool_visD){
	    		vector<string> sub_str;
   				const string full_str = string(argv[2]);
  				char del = '.';
   				split(full_str,del,sub_str);
   				string new_name = sub_str[0]+"_mask.png";
   				imwrite(new_name,mask);
	    	}

	    	//-- Getting numerical data
	    	vector<double> shapes_data = get_shapes(cc,mask);
	    	Mat hue_data = get_color(adjImage, mask);

		    //-- Write shapes to file
	    	string name_shape= string(argv[3]);
	    	ofstream shape_file;
	    	shape_file.open(name_shape.c_str(),ios_base::app);
	    	shape_file << argv[2] << " ";
	    	for(int i=0;i<20;i++){
	    		shape_file << shapes_data[i];
	    		if(i != 19){
	    			shape_file << " ";
	    		}
	    	}
	    	shape_file << endl;
	    	shape_file.close();

	    	//-- Write color to file
	    	string name_hue= string(argv[4]);
	    	ofstream hue_file;
	    	hue_file.open(name_hue.c_str(),ios_base::app);
	    	hue_file << argv[2] << " ";
	    	for(int i=0;i<180;i++){
	    		hue_file << hue_data.at<float>(i,0) << " ";
	    	}
	    	hue_file << endl;
	    	hue_file.close();
		}

	}
	else if(bool_avgImgs){
		if(argc != 2){
			cout << "Using mode AVG_IMGS requires only that a list of images to be averaged is piped in" << endl;
		}else{
			 //-- Taking list of pictures that are piped in and averaging them
					string line;
					Mat avg;
					vector<Mat> avg_bgr(3);
					int counter = 0;
					Mat adjImage1;
					while(cin) {
						if(getline(cin,line)) {
							if(counter == 0){
					    		avg=imread(line);
						    	cvtColor(avg, adjImage1, CV_BGRA2BGR);
						    	adjImage1.convertTo(adjImage1, CV_64F);
					   			split(adjImage1,avg_bgr);
					    		counter++;
					    	}else{
					        	Mat inputImage = imread(line);
						    	cvtColor(inputImage, adjImage1, CV_BGRA2BGR);
						    	adjImage1.convertTo(adjImage1, CV_64F);
					    		vector<Mat> in_bgr(3);
				    			split(adjImage1,in_bgr);
				    			avg_bgr[0] = (avg_bgr[0]+in_bgr[0]);
				    			avg_bgr[1] = (avg_bgr[1]+in_bgr[1]);
				    			avg_bgr[2] = (avg_bgr[2]+in_bgr[2]);
					        	counter++;
					    	}
					    }
					}
					avg_bgr[0] = (avg_bgr[0])/counter;
					avg_bgr[1] = (avg_bgr[1])/counter;
					avg_bgr[2] = (avg_bgr[2])/counter;
					Mat adjImage;
					merge(avg_bgr,adjImage);
					adjImage.convertTo(adjImage, CV_64F);

					//-- Writing out averaged image
					imwrite("average_images.png",adjImage);
		}
	}
	else if(bool_hyper || bool_hyperD){
			if(argc != 5){
				cout << "Using mode HYPER requires input in this order: hyperspectral_dir/ hyper_shapes.txt hyper_color.txt" << endl;
			}else{
				string name_hyper_fail= "failed_images.txt";
				ofstream hyper_file_fail;
				string line = argv[2];
				cout << line << endl;
				try{
					//-- Getting important images
					Mat m705 = imread(line+"35_0_0.png",CV_LOAD_IMAGE_GRAYSCALE);
					m705.convertTo(m705, CV_64F);
					Mat m750 = imread(line+"45_0_0.png",CV_LOAD_IMAGE_GRAYSCALE);
					m750.convertTo(m750, CV_64F);
					Mat m1056 = imread(line+"108_0_0.png",CV_LOAD_IMAGE_GRAYSCALE);
					m1056.convertTo(m1056, CV_64F);
					Mat m1151 = imread(line+"128_0_0.png",CV_LOAD_IMAGE_GRAYSCALE);
					m1151.convertTo(m1151, CV_64F);

					Mat img;
					//-- Threshold and ROI for whole plant
					Mat mask_total;
					img = ((m750+1)-(m705+1))/((m750+1)+(m705+1));
					inRange(img,0.18,1.5,mask_total);

					Mat m57 = imread(line+"57_0_0.png");
					m57.convertTo(m57, CV_BGR2GRAY);
					Mat m57_thresh;
					inRange(m57,0,55,m57_thresh);
					Mat pot_mask;
					bitwise_and(m57_thresh,mask_total,pot_mask);
					Mat mask_total1 = mask_total - pot_mask;
					Mat kept_mask_hyp_total;
					vector<Point> cc_total = keep_roi(mask_total1,Point(55,123),Point(270,357),kept_mask_hyp_total);

					//-- Threshold and ROI for stem
					Mat mask_stem;
					img = (m1056+1)/(m1151+1);
					inRange(img,1.1,5,mask_stem);
					Mat stem_and;
					bitwise_and(kept_mask_hyp_total,mask_stem,stem_and);
					Mat kept_mask_hyp_stem;
					vector<Point> cc_stem = keep_roi(stem_and,Point(55,123),Point(270,357),kept_mask_hyp_stem);

					//-- Threshold and ROI for leaves
					Mat mask_leaves = kept_mask_hyp_total-kept_mask_hyp_stem;
					Mat kept_mask_hyp_leaves;
					vector<Point> cc_leaves = keep_roi(mask_leaves,Point(55,123),Point(270,357),kept_mask_hyp_leaves);

					//-- Getting and writing shapes data
					vector<double> shapes_total = get_shapes(cc_total,kept_mask_hyp_total);
					vector<double> shapes_stem = get_shapes(cc_stem,kept_mask_hyp_stem);
					vector<double> shapes_leaves = get_shapes(cc_leaves,kept_mask_hyp_leaves);

					string name_shape= string(argv[3]);
					ofstream shape_file;
					shape_file.open(name_shape.c_str(),ios_base::app);

					shape_file << argv[2] << " total ";
					for(int i=0;i<20;i++){
						shape_file << shapes_total[i];
						if(i != 19){
							shape_file << " ";
						}
					}
					shape_file << endl;

					shape_file << argv[2] << " stem ";
					for(int i=0;i<20;i++){
						shape_file << shapes_stem[i];
						if(i != 19){
							shape_file << " ";
						}
					}
					shape_file << endl;

					shape_file << argv[2] << " leaves ";
					for(int i=0;i<20;i++){
						shape_file << shapes_leaves[i];
						if(i != 19){
							shape_file << " ";
						}
					}
					shape_file << endl;
					shape_file.close();

					if(bool_hyperD){
		   				string new_name;
		   				new_name = line+"total_mask.png";
		   				imwrite(new_name,kept_mask_hyp_total);
		   				new_name = line+"stem_mask.png";
		   				imwrite(new_name,kept_mask_hyp_stem);
		   				new_name = line+"leaves_mask.png";
		   				imwrite(new_name,kept_mask_hyp_leaves);
					}

					//-- Looping over all the wavelengths and writing out histogram
					string name_hyper_color= string(argv[4]);
					ofstream hyper_file_color;
					hyper_file_color.open(name_hyper_color.c_str(),ios_base::app);
					for(int i=2;i<245;i++){
						stringstream ss;
					  	ss << i;
					   	string str = ss.str();
					   	Mat in_image;
						in_image = imread(line+str+"_0_0.png",CV_LOAD_IMAGE_GRAYSCALE);
					   	Mat hyper_data_total = get_gray(in_image, kept_mask_hyp_total);
					   	Mat hyper_data_stem = get_gray(in_image, kept_mask_hyp_stem);
					   	Mat hyper_data_leaves = get_gray(in_image, kept_mask_hyp_leaves);

					  	//-- Write total plant histogram
					   	hyper_file_color << line+str+"_0_0.png" << " total ";
						for(int i=0;i<255;i++){
						   	hyper_file_color << hyper_data_total.at<float>(i,0) << " ";
						}
						hyper_file_color << endl;

						//-- Write stem only histogram
						hyper_file_color << line+str+"_0_0.png" << " stem ";
						for(int i=0;i<255;i++){
							hyper_file_color << hyper_data_stem.at<float>(i,0) << " ";
						}
						hyper_file_color << endl;

						//-- Write leaves only histogram
						hyper_file_color << line+str+"_0_0.png" << " leaves ";
						for(int i=0;i<255;i++){
							hyper_file_color << hyper_data_leaves.at<float>(i,0) << " ";
						}
						hyper_file_color << endl;
					}
					hyper_file_color.close();
				}
				catch (Exception& e) {
					hyper_file_fail.open(name_hyper_fail.c_str(),ios_base::app);
					hyper_file_fail << line << endl;
					hyper_file_fail.close();
				}

			}
	}
	else if(mode == "-h" || mode == "--help"){
		cout << "DESCRIPTION:" << endl << "\tThis program is for segmenting and measuring plants from the phenotyping facility in University of Nebraska - Lincoln" << endl << endl;
		cout << "USAGE:" << endl << "\tThere are five modes of use (VIS, VIS_DEBUG, HYPER, NIR, and AVG_IMGS). Depending on what is chosen, the required inputs change" << endl << endl;
		cout << "SYNOPSIS:" << endl << "\t./PhenotyperCV [MODE] [INPUTS]" << endl << endl;
		cout << "MODES:"<< endl;
		cout << "\t\e[1mVIS\e[0m - Segment and measure plant in RGB images" << endl << "\t" << "Example: ./PhenotyperCV VIS input_image.png shapes.txt color.txt"<< endl<<endl;
		cout << "\t\e[1mVIS_DEBUG\e[0m - Same as VIS but will output the mask in the same directory as the input image" << endl << "\t" << "Example: ./PhenotyperCV VIS_DEBUG input_image.png shapes.txt color.txt"<< endl<<endl;
		cout << "\t\e[1mHYPER\e[0m - Segment plant into total, stem and leaves then measure hyperspectral data with masks" << endl << "\t" << "Example: ./PhenotyperCV HYPER hyperspectral_dir/ hyper_shapes.txt hyper_color.txt"<< endl<<endl;
		cout << "\t\e[1mHYPER_DEBUG\e[0m - Same as HYPER but will output the masks in the same directory" << endl << "\t" << "Example: ./PhenotyperCV HYPER_DEBUG hyperspectral_dir/ hyper_shapes.txt hyper_color.txt"<< endl<<endl;
		cout << "\t\e[1mAVG_IMGS\e[0m - takes list of input images to be averaged and outputs average_images.png" << endl << "\t" << "Example: cat Images/SnapshotInfo.csv | grep Fm000Z | grep VIS_SV | awk -F'[;,]' '{print \"Images/snapshot\"$2\"/\"$12\".png\"}' | ./PhenotyperCV AVG_IMGS"<< endl << endl;
		cout << "PARALLELIZATION:" << endl;
		cout << "\tHyperspectral:\n\techo Images/*/Hyp_SV_90/ | sed 's/ /\\n/g' | grep -v \"Empty\" | xargs -I{} -P8 ./PhenotyperCV HYPER_DEBUG {} hyper_shapes_local_11-5.txt hyper_color_local_11-5.txt" << endl;
		cout << "\tVIS:\n\tfind Images/ -name \"*.png\" | grep Vis_SV | xargs -I{} -P8 ./UNL_VIS_IA VIS_DEBUG {} shapes.txt color.txt" << endl;

	}
	else{
		cout << "First argument must be either VIS, VIS_DEBUG, HYPER, HYPER_DEBUG or AVG_IMGS" << endl;
		cout << "Use  ./PhenotyperCV --help  for more information" << endl;
	}

	return 0;
}

/*
namedWindow("Image",WINDOW_NORMAL);
        	    resizeWindow("Image",800,800);
        	    imshow("Image", b_blur);
waitKey(0);
*/
