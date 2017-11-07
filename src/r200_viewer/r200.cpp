#include "r200.h"

void normalize_depth_to_rgb(uint8_t rgb_image[640 * 480 * 3], const uint16_t depth_image[], int width, int height)
{
	for (int i = 0; i < width * height; ++i)
	{
		if (uint16_t d = depth_image[i])
		{

			uint8_t v = (d * 255 / std::numeric_limits<uint16_t>::max());
			//cout << d << " " << std::numeric_limits<uint16_t>::max() << " " << d * 255 / std::numeric_limits<uint16_t>::max() <<  " " << v << endl;

			rgb_image[i * 3 + 0] = (255 - v);		//B
			rgb_image[i * 3 + 1] = (255 - v);		//G
			rgb_image[i * 3 + 2] = (255 - v);		//R
		}
		else
		{
			rgb_image[i * 3 + 0] = 0;
			rgb_image[i * 3 + 1] = 0;
			rgb_image[i * 3 + 2] = 0;
		}
	}
}
void normalize_depth_to_rgb2(uint8_t rgb_image[640 * 480 * 3], const uint16_t depth_image[], int width, int height)
{
	static uint32_t histogram[0x10000];
	memset(histogram, 0, sizeof(histogram));

	for (int i = 0; i < width*height; ++i) ++histogram[depth_image[i]];
	for (int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i - 1]; // Build a cumulative histogram for the indices in [1,0xFFFF]
	for (int i = 0; i < width*height; ++i)
	{
		if (uint16_t d = depth_image[i])
		{
			int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
			rgb_image[i * 3 + 0] = 255 - f;
			rgb_image[i * 3 + 1] = 0;
			rgb_image[i * 3 + 2] = f;
		}
		else
		{
			rgb_image[i * 3 + 0] = 20;
			rgb_image[i * 3 + 1] = 5;
			rgb_image[i * 3 + 2] = 0;
		}
	}
}

float deg2rad(float degree)
{
	float radian;
	radian = degree * PI / 180.0;

	return radian;
}



r200_buffer::r200_buffer()
{
}
r200_buffer::~r200_buffer()
{


}
const void* r200_buffer::show(rs::device & dev, rs::stream stream)
{
	const void * data;
	if (!dev.is_stream_enabled(stream)) return data;
	assert(dev.is_stream_enabled(stream));
	const int timestamp = dev.get_frame_timestamp(stream);
	if (timestamp != last_timestamp)
	{
		data = upload(dev.get_frame_data(stream), dev.get_stream_width(stream), dev.get_stream_height(stream), dev.get_stream_format(stream));
		last_timestamp = timestamp;

		++num_frames;
		if (timestamp >= next_time)
		{
			fps = num_frames;
			num_frames = 0;
			next_time += 1000;
		}
	}

	return data;
}
const void * r200_buffer::upload(const void * data, int width, int height, rs::format format)
{
	switch (format)
	{
	case rs::format::any:
		throw std::runtime_error("not a valid format");
	case rs::format::z16:	// Disparity
	case rs::format::disparity16:
		rgb.resize(width * height * 3);
		normalize_depth_to_rgb2(rgb.data(), reinterpret_cast<const uint16_t *>(data), width, height);
		return rgb.data();
	case rs::format::bgr8:	// BGR
		return data;
	}


}