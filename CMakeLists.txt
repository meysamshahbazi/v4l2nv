cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(v4l2nv VERSION 0.0.0 LANGUAGES C CXX)

find_package(Threads)
find_package(CUDA REQUIRED)
# find_package(CUDNN REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(TensorRT REQUIRED)

set(CUDA_PATH /usr/local/cuda-10.2)
set(TEGRA_ARMABI aarch64-linux-gnu)



set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)


# Define project name


include_directories(
    inc
	# -I"$(TOP_DIR)/include/libjpeg-8b" \
	# -I"$(ALGO_CUDA_DIR)" \
	# -I"$(ALGO_TRT_DIR)" \
	${CUDA_PATH}/include
	/usr/include/${TEGRA_ARMABI}
	/usr/include/libdrm
	# -I"$(TARGET_ROOTFS)/usr/include/opencv4"
        # /usr/include/gstreamer-1.0 
        # /usr/include/glib-2.0 
        # /usr/lib/aarch64-linux-gnu/glib-2.0/include 
        # /usr/lib/x86_64-linux-gnu/glib-2.0/include 
        )

# set(CLASS_DIR 	:= $(TOP_DIR)/samples/common/classes )

file(GLOB CLASS_SRC "classes/*.cpp")

link_directories(	
            ${CUDA_PATH}/lib64
            /usr/lib/${TEGRA_ARMABI}
            /usr/lib/${TEGRA_ARMABI}/tegra
        )

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_72,code=sm_72)


SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------

cuda_add_executable(${PROJECT_NAME} main.cpp ${CLASS_SRC})

# message(WARNING ${CMAKE_MFC_FLAG})
set(CMAKE_CXX_STANDARD 14)
add_compile_options(${CMAKE_MFC_FLAG} )
# add_executable(${PROJECT_NAME} main.c cssl.c)
target_link_libraries(${PROJECT_NAME}  	-lpthread -lv4l2 -lEGL -lGLESv2 -lX11 -lnvbuf_utils -lnvjpeg -lnvosd -ldrm -lcuda -lcudart )

# target_include_directories(${PROJECT_NAME} PUBLIC ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})






# target_include_directories(${PROJECT_NAME} PUBLIC )
# cuda_add_executable(${PROJECT_NAME} utils.cu )
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(${PROJECT_NAME}  ${CUDNN_LIBRARIES}  ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )
message(${OpenCV_LIBS})