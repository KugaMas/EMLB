cmake_minimum_required(VERSION 3.5.1)
project(EMLB)

#最后发布加入这句话，编译用release模式，比默认的debug模式速度慢很多 
# set(CMAKE_BUILD_TYPE "Release")
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/scripts/utils)

# # find_package(Boost COMPONENTS thread system REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# # include_directories(${PROJECT_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

pybind11_add_module(cdn_utils ${PROJECT_SOURCE_DIR}/scripts/src/edn.cpp)

# target_link_libraries(cdn_utils PRIVATE
#     ${Boost_LIBRARIES}
# )
