# cmake needs this line
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project name
project(PR)

# set(OpenCV_DIR "F:\\ws\\Programming\\ThirdParties\\opencv\\opencv33\\build")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

# If in Linux, try this:
find_package(OpenCV REQUIRED)
# Using dlib as the thirdparty for face detection
find_package(dlib REQUIRED)

# # If in Windows, try this:
# set(OpenCV_INCLUDE_DIRS F:\\ws\\Programming\\ThirdParties\\opencv\\opencv33\\build\\include)
# set(OpenCV_LIBS F:\\ws\\Programming\\ThirdParties\\opencv\\opencv33\\build\\x64\\vc14\\lib\\opencv_world330d.lib)


# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
# Add dlib headers location to your include paths
include_directories(${dlib_INCLUDE_DIRS})

# Declare the executable target built from your sources
add_executable(parzen
        src/parzen.cpp
        src/DataLoader.cpp
        src/DataLoader.h
        src/DataStructure.h)

# Link your application with OpenCV libraries
target_link_libraries(parzen ${OpenCV_LIBS})


add_executable(face
        src/face.cpp
        src/DataLoader.cpp
        src/DataLoader.h
        src/DataStructure.h)

target_link_libraries(face ${OpenCV_LIBS} ${dlib_LIBRARIES})

# Declare the executable target built from your sources
#add_executable(knn
#        src/knn.cpp
#        src/DataLoader.cpp
#        src/DataLoader.h
#        src/DataStructure.h)

# Link your application with OpenCV libraries
#target_link_libraries(knn ${OpenCV_LIBS})