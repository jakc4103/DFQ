cmake_minimum_required(VERSION 2.8.12)
project(dfq)
include_directories(/home/jakc4103/Documents/ncnn/src)
include_directories(/home/jakc4103/Documents/ncnn/build/src)

#openmp
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#ncnn
set(NCNN_LIBS /home/jakc4103/Documents/ncnn/build/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS /home/jakc4103/Documents/ncnn/build/install/include)
include_directories(${NCNN_INCLUDE_DIRS})

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(inference_cls inference_cls.cpp)
target_link_libraries(inference_cls ${NCNN_LIBS})
target_link_libraries( inference_cls ${OpenCV_LIBS} )