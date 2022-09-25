#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

class Histogram {
public:
	Mat calc_histogram(Mat scr) {
		Mat hist;
		hist = Mat::zeros(256, 1, CV_32F);
		scr.convertTo(scr, CV_32F);
		double value = 0;
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				value = scr.at<float>(i, j);
				hist.at<float>(value) = hist.at<float>(value) + 1;
			}
		}
		return hist;
	}

	void plot_histogram(Mat histogram) {
		Mat histogram_image(400, 512, CV_8UC3, Scalar(0, 0, 0)); //The type Scalar is widely used in OpenCV to pass pixel values
		Mat normalized_histogram;
		normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());

		for (int i = 0; i < 256; i++)
		{
			rectangle(histogram_image, Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)),
				Point(2 * (i + 1), histogram_image.rows), Scalar(255, 0, 0));
		}

		namedWindow("Histogramoforiginalimage", WINDOW_NORMAL);
		imshow("Histogramoforiginalimage", histogram_image);
	}

	void plot_histogram_eq(Mat histogram) {
		Mat histogram_image(400, 512, CV_8UC3, Scalar(0, 0, 0));
		Mat normalized_histogram;
		normalize(histogram, normalized_histogram, 0, 400, NORM_MINMAX, -1, Mat());

		for (int i = 0; i < 256; i++)
		{
			rectangle(histogram_image, Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)),
				Point(2 * (i + 1), histogram_image.rows), Scalar(255, 0, 0));
		}

		namedWindow("Histogramofequlizedimage", WINDOW_NORMAL);
		imshow("Histogramofequilizedimage", histogram_image);
	}

};

void main() {
	string name;
	cout << "Paste The Path To Get the Histogram :\n";
	cin >> name;
	Mat img;
	img = imread(name);
	
	// Check for failure
	if (img.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
	}

	//Convert the image from BGR to YCrCb color space
	Mat hist_equalized_image;
	cvtColor(img, hist_equalized_image, COLOR_BGR2YCrCb);

	//Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
	vector<Mat> vec_channels;
	split(hist_equalized_image, vec_channels);

	//Equalize the histogram of only the Y channel 
	equalizeHist(vec_channels[0], vec_channels[0]);

	//Merge 3 channels in the vector to form the color image in YCrCB color space.
	merge(vec_channels, hist_equalized_image);

	//Convert the histogram equalized image from YCrCb to BGR color space again
	cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

	//Define the names of windows
	String windowNameOfOriginalImage = "Original Image";
	String windowNameOfHistogramEqualized = "Histogram Equalized Color Image";

	// Create windows with the above names
	namedWindow(windowNameOfOriginalImage, WINDOW_NORMAL);
	namedWindow(windowNameOfHistogramEqualized, WINDOW_NORMAL);

	// Show images inside the created windows.
	imshow(windowNameOfOriginalImage, img);
	imshow(windowNameOfHistogramEqualized, hist_equalized_image);


	Histogram H1;
	Mat hist = H1.calc_histogram(img);
	H1.plot_histogram(hist);

	Histogram H2;
	Mat hist2 = H2.calc_histogram(hist_equalized_image);
	H2.plot_histogram_eq(hist2);

	waitKey();

	destroyAllWindows(); //Destroy all opened windows

	waitKey();
}
