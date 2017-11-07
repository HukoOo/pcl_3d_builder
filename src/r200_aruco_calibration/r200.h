#pragma once

#include <librealsense/rs.hpp>
// opencv
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#define WIDTH 640
#define HEIGHT 480
#define PI 3.1415926

class r200_buffer
{
private:


public:
	int last_timestamp;
	int fps, num_frames, next_time;
	std::vector<uint8_t> rgb;
	cv::Mat image;

	// r200 parameters
	rs::intrinsics depth_intrin;
	rs::intrinsics color_intrin;
	rs::extrinsics depth_to_color;

	r200_buffer();
	~r200_buffer();

	const void* show(rs::device & dev, rs::stream stream);
	const void * upload(const void * data, int width, int height, rs::format format);

};