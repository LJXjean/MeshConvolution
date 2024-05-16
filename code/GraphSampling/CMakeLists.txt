cmake_minimum_required(VERSION 3.5)
project(GraphSampling)

set(CMAKE_CXX_STANDARD 14)

# Disable warnings for unsafe functions
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# Use environment variables to find ZLIB
set(ZLIB_INCLUDE_DIR $ENV{ZLIB_INCLUDE_DIR})
set(ZLIB_LIBRARY_RELEASE $ENV{ZLIB_LIBRARY_RELEASE})
set(ZLIB_LIBRARY_DEBUG $ENV{ZLIB_LIBRARY_DEBUG})

# Find ZLIB
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIR})


# Set OpenCV directory
set(OpenCV_DIR "D:/Microsoft Edge Download/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(GraphSampling main.cpp mdSArray.h mdVector.h mdVector.inl meshLoader.h meshPooler.h meshPooler_visualizer_new.h meshCNN.h cnpy/cnpy.cpp cnpy/cnpy.h)

target_link_libraries(GraphSampling debug ${ZLIB_LIBRARY_DEBUG} optimized ${ZLIB_LIBRARY_RELEASE})

target_link_libraries(GraphSampling debug ${ZLIB_LIBRARY_DEBUG} optimized ${ZLIB_LIBRARY_RELEASE} ${OpenCV_LIBS})
