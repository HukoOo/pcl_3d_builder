cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(aruco_pose_estimation)

#PCL 1.8
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#OpenCV 3.2
find_package(OpenCV REQUIRED)

#librealsense > 1.1.0
#find_package(librealsense)
set(librealsense_DIR "D:/Dev/Intel/librealsense/build_x64_vc14/install")

file(GLOB r200_viewer_SRC
"src/r200_viewer/*.h"
"src/r200_viewer/*.cpp"
)

include_directories(
    ${librealsense_DIR}/include
    )

add_executable( r200_viewer ${r200_viewer_SRC})

target_link_libraries (r200_viewer
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${librealsense_DIR}/lib/realsense.lib
    )
