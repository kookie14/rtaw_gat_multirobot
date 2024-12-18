cmake_minimum_required(VERSION 3.1)
get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(${PROJECT})
add_compile_options(-std=c++17)

## System dependencies are found with CMake's conventions
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch)

## System dependencies are found with CMake's conventions
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
file(GLOB CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/**/*.cpp)

set(EXTERNAL_LIBS ${Boost_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${GLOG_LIBRARY}  Matplot++::matplot)
# Set include directories
include_directories( 
    ${EIGEN3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    include
)
# Set libraries
add_library(${PROJECT}
    ${CPP_SOURCES}
)

# add_executable(tm_rds_exe src/main.cpp 
#     ${CPP_SOURCES}
# )

# target_link_libraries(tm_rds_exe ${EXTERNAL_LIBS})
# set_property(TARGET tm_rds_exe PROPERTY CXX_STANDARD 17)

add_executable(rtaw_allocation_train_exe src/rtaw_allocation_train.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(rtaw_allocation_train_exe ${EXTERNAL_LIBS})
set_property(TARGET rtaw_allocation_train_exe PROPERTY CXX_STANDARD 17)


add_executable(gnn_allocation_train_exe src/gnn_allocation_train.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(gnn_allocation_train_exe ${EXTERNAL_LIBS})
set_property(TARGET gnn_allocation_train_exe PROPERTY CXX_STANDARD 17)

add_executable(sac_gnn_allocation_train_exe src/sac_gnn_allocation_train.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(sac_gnn_allocation_train_exe ${EXTERNAL_LIBS})
set_property(TARGET sac_gnn_allocation_train_exe PROPERTY CXX_STANDARD 17)

add_executable(astar_allocation_test_exe src/astar_allocation_test.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(astar_allocation_test_exe ${EXTERNAL_LIBS})
set_property(TARGET astar_allocation_test_exe PROPERTY CXX_STANDARD 17)

add_executable(rtaw_allocation_test_exe src/rtaw_allocation_test.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(rtaw_allocation_test_exe ${EXTERNAL_LIBS})
set_property(TARGET rtaw_allocation_test_exe PROPERTY CXX_STANDARD 17)

add_executable(gnn_allocation_test_exe src/gnn_allocation_test.cpp 
    ${CPP_SOURCES}
)

target_link_libraries(gnn_allocation_test_exe ${EXTERNAL_LIBS})
set_property(TARGET gnn_allocation_test_exe PROPERTY CXX_STANDARD 17)