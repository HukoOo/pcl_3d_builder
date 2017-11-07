#include <librealsense/rs.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace rs;
using namespace cv;

// Window size and frame rate
int const INPUT_WIDTH = 640;
int const INPUT_HEIGHT = 480;
int const FRAMERATE = 60;

// Named windows
char* const WINDOW_DEPTH = "Depth Image";
char* const WINDOW_RGB = "RGB Image";


rs::context      _rs_ctx;
rs::device&      _rs_camera = *_rs_ctx.get_device(0);
rs::intrinsics   _depth_intrin;
rs::intrinsics  _color_intrin;
bool         _loop = true;




namespace {
	const char* about = "Pose estimation using a ArUco Planar Grid board";
	const char* keys =
		"{w        |       | Number of squares in X direction }"
		"{h        |       | Number of squares in Y direction }"
		"{l        |       | Marker side lenght (in pixels) }"
		"{s        |       | Separation between two consecutive markers in the grid (in pixels)}"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		"{c        |       | Output file with calibrated camera parameters }"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{ci       | 0     | Camera id if input doesnt come from video (-v) }"
		"{dp       |       | File of marker detector parameters }"
		"{rs       |       | Apply refind strategy }"
		"{r        |       | show rejected candidates too }";
}

/**
*/
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}



/**
*/
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["doCornerRefinement"] >> params->doCornerRefinement;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}


void depth_to_cloud(rs::device& _rs_camera, cv::Mat& rgb, cv::Mat& u16, cv::Mat& u8)
{
	// Retrieve camera parameters for mapping between depth and color
	rs::intrinsics depth_intrin = _rs_camera.get_stream_intrinsics(rs::stream::depth_aligned_to_color);
	rs::intrinsics color_intrin = _rs_camera.get_stream_intrinsics(rs::stream::color);
	rs::extrinsics depth_to_color = _rs_camera.get_extrinsics(rs::stream::depth_aligned_to_color, rs::stream::color);

	/// return  depth in meters corresponding to a depth value of 1
	float scale = _rs_camera.get_depth_scale();
	static const float nan = std::numeric_limits<float>::quiet_NaN();
	float max_distance = 10;

	uint16_t* u16_Ptr = (uint16_t*)u16.data;
	uint8_t* u8_Ptr = (uint8_t*)u8.data;
	uint8_t* pixelPtr = (uint8_t*)rgb.data;

	int cn = u8.channels();

	static uint32_t histogram[0x10000];
	memset(histogram, 0, sizeof(histogram));

	for (int j = 0; j < INPUT_HEIGHT; j++)
	{
		for (int i = 0; i < INPUT_WIDTH; i++)
		{
			++histogram[u16_Ptr[j *  INPUT_WIDTH + i]];
		}
	}
	for (int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i - 1]; // Build a cumulative histogram for the indices in [1,0xFFFF]

	for (int j = 0; j < INPUT_HEIGHT; j++)
	{
		uint idx = j * INPUT_WIDTH - 1;
		for (int i = 0; i < INPUT_WIDTH; i++)
		{
			idx++;
			if (uint16_t d = u16_Ptr[j *  INPUT_WIDTH + i])
			{
				float depth_in_meters = d * scale;		// mm to m

														//normalize 
				int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location

				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 0] = 255 - f;
				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 1] = 0;
				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 2] = f;

				//cloud
				// Map from pixel coordinates in the depth image to camera co-ordinates
				// TODO : transform from camera coordinates to the robot coordinates
				rs::float2 depth_pixel = { (float)i, (float)j };
				rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
				rs::float3 color_point = depth_to_color.transform(depth_point);
				rs::float2 color_pixel = color_intrin.project(color_point);
				const int cx = (int)std::round(color_pixel.x), cy = (int)std::round(color_pixel.y);

				//const int cx = dx, cy = dy;
				int red = 0, green = 0, blue = 0;
				if (cx < 0 || cy < 0 || cx > INPUT_WIDTH || cy > INPUT_HEIGHT)
				{
					red = 255; green = 255; blue = 255;
				}
				else
				{
					int pos = (cy * INPUT_WIDTH + cx) * 3;
					red = pixelPtr[cy*INPUT_WIDTH*cn + cx*cn + 0];
					green = pixelPtr[cy*INPUT_WIDTH*cn + cx*cn + 1];
					blue = pixelPtr[cy*INPUT_WIDTH*cn + cx*cn + 2];
				}
				if (d == 0 || depth_point.z > max_distance)
				{
					//_cloud->points[idx].x = _cloud->points[idx].y = _cloud->points[idx].z = (float)nan;
					continue;
				}
				else
				{
				}
			}
			else
			{
				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 0] = 0;
				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 1] = 0;
				u8_Ptr[j*INPUT_WIDTH*cn + i*cn + 2] = 0;
			}
		}
	}

}

bool initialize_streaming()
{
	bool success = false;
	if (_rs_ctx.get_device_count() > 0)
	{
		_rs_camera.enable_stream(rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::rgb8, FRAMERATE);
		_rs_camera.enable_stream(rs::stream::depth, INPUT_WIDTH, INPUT_HEIGHT, rs::format::z16, FRAMERATE);
		_rs_camera.start();
		//_rs_camera.set_option(rs::option::r200_lr_auto_exposure_enabled, 1);

		success = true;
	}
	return success;
}

void setup_windows()
{
	cv::namedWindow(WINDOW_DEPTH, 1);
	cv::namedWindow(WINDOW_RGB, 1);
}

int main(int argc, char *argv[]) try
{
	/////////////////////////////////////////////     Aruco Init
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 7) {
		parser.printMessage();
		return 0;
	}

	int markersX = parser.get<int>("w");
	int markersY = parser.get<int>("h");
	float markerLength = parser.get<float>("l");
	float markerSeparation = parser.get<float>("s");
	int dictionaryId = parser.get<int>("d");
	bool showRejected = parser.has("r");
	bool refindStrategy = parser.has("rs");

	Mat camMatrix, distCoeffs;
	if (parser.has("c")) 
	{
		bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
		if (!readOk) {
			cerr << "Invalid camera file" << endl;
			return 0;
		}
	}

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	if (parser.has("dp")) {
		bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
		if (!readOk) {
			cerr << "Invalid detector parameters file" << endl;
			return 0;
		}
	}
	detectorParams->doCornerRefinement = true; // do corner refinement in markers

	String video;
	if (parser.has("v")) {
		video = parser.get<String>("v");
	}

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	float axisLength = 0.5f * ((float)min(markersX, markersY) * (markerLength + markerSeparation) +
		markerSeparation);

	// create board object
	Ptr<aruco::GridBoard> gridboard =
		aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);
	Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();

	double totalTime = 0;
	int totalIterations = 0;

	/////////////////////////////////////////////     Realsense Init
	rs::log_to_console(rs::log_severity::warn);

	if (!initialize_streaming())
	{
		std::cout << "Unable to locate a camera" << std::endl;
		rs::log_to_console(rs::log_severity::fatal);
		return EXIT_FAILURE;
	}

	setup_windows();


	int cnt = 0;
	while (_rs_camera.is_streaming())
	{
		double tick = (double)getTickCount();

		// get the image data
		if (_rs_camera.is_streaming())
			_rs_camera.wait_for_frames();

		_depth_intrin = _rs_camera.get_stream_intrinsics(rs::stream::depth);
		_color_intrin = _rs_camera.get_stream_intrinsics(rs::stream::color);
		
		// Create depth image
		cv::Mat depth16(_depth_intrin.height,
			_depth_intrin.width,
			CV_16U,
			(uchar *)_rs_camera.get_frame_data(rs::stream::depth_aligned_to_color));

		// Create color image
		cv::Mat image(_color_intrin.height,
			_color_intrin.width,
			CV_8UC3,
			(uchar *)_rs_camera.get_frame_data(rs::stream::color));
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		// < 800
		cv::Mat depth8u = cv::Mat::zeros(_depth_intrin.height, _depth_intrin.width, CV_8UC3);
		depth_to_cloud(_rs_camera, image, depth16, depth8u);

				
		/////////////////////////////////////////////    algorithm
		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;
		Vec3d rvec, tvec;

		// detect markers
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

		// refind strategy to detect more markers
		if (refindStrategy)
			aruco::refineDetectedMarkers(image, board, corners, ids, rejected, camMatrix,
				distCoeffs);

		// estimate board pose
		int markersOfBoardDetected = 0;
		if (ids.size() > 0)
			markersOfBoardDetected =
			aruco::estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs, rvec, tvec);

		double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
		totalTime += currentTime;
		totalIterations++;
		if (totalIterations % 30 == 0) {
			cout << "Detection Time = " << currentTime * 1000 << " ms "
				<< "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
		}

		// draw results
		cv::Mat imageCopy;
		image.copyTo(imageCopy);
		if (ids.size() > 0) {
			aruco::drawDetectedMarkers(imageCopy, corners, ids);
		}

		if (showRejected && rejected.size() > 0)
			aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

		if (markersOfBoardDetected > 0)
			aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

		imshow("out", imageCopy);
		char key = (char)waitKey(1);
		if (key == 27) break;

	}

	_rs_camera.stop();
	cv::destroyAllWindows();


	return EXIT_SUCCESS;

}
catch (const rs::error & e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}