cmake_minimum_required(VERSION 3.16)
project(slam_algorithm)
# Check C++11 or C++0x support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Debug")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV 4.1 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# Eigen3
find_package(Eigen3 3.1.0 REQUIRED)
# Sophus
find_package(Sophus REQUIRED)
# Ceres
find_package(Ceres REQUIRED)
# G2O
find_package(G2O REQUIRED)

# csparse
find_package(CSparse REQUIRED)


set(Sophus_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/Sophus")



include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${CERES_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
)
SET(G2O_LIBS g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension)
        #link_libraries(${PROJECT_NAME}
#        ${OpenCV_LIBS}
#        ${EIGEN3_LIBS}
#        ${CERES_LIBRARIES})


add_executable(gauss_newton
        Optimization/gauss_newton.cpp
        CMakeLists.txt)
target_link_libraries(gauss_newton
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )

add_executable(pnp
        BundleAdjustment/pnp.cpp
        BundleAdjustment/My_EPNP.cpp
        CMakeLists.txt)
target_link_libraries(pnp
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${CERES_LIBRARIES}
        ${G2O_LIBS}
        ${CSPARSE_LIBRARY}
        )

add_executable(pose_2d2d
        BundleAdjustment/pose_2d2d.cpp
        CMakeLists.txt)
target_link_libraries(pose_2d2d
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )

add_executable(image_stitch Features/image_stitch.cpp)
target_link_libraries(image_stitch
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )

add_executable(triangulator Stereo/triangulator.cpp)
target_link_libraries(triangulator
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )

add_executable(stereo_calibrate Stereo/stereo_calibrate.cpp)
target_link_libraries(stereo_calibrate
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )


add_executable(stereo Stereo/stereo.h Stereo/stereo.cpp Stereo/fundamental_essential_test.cpp)
target_link_libraries(stereo
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )
