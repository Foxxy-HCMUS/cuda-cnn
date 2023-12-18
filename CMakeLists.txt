# cmake_minimum_required (VERSION 3.0)
# project (myproject)
# include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# # 查找当前目录下的所有源文件
# # 并将名称保存到 DIR_SRCS 变量
# aux_source_directory(. DIR_SRCS)
# # 添加 math 子目录
# add_subdirectory(src)
# # 指定生成目标 
# add_executable(demo ${DIR_SRCS})
# # 添加链接库
# target_link_libraries(demo MiniDNNLib)

cmake_minimum_required (VERSION 3.8)  # 3.8 is required for CUDA language support
project (myproject LANGUAGES CXX CUDA)  # Enable CUDA as a language
include_directories(${CUDA_INCLUDE_DIRS})
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen-3.4.0")

# Find all source files in the current directory
file(GLOB DIR_SRCS *.cc *.cu)

# Add src subdirectory
add_subdirectory(src)

# Specify the target executable
add_executable(demo ${DIR_SRCS})

# Link the MiniDNNLib library
target_link_libraries(demo MiniDNNLib)

# Link CUDA libraries
find_package(CUDA REQUIRED)

target_link_libraries(demo ${CUDA_LIBRARIES})


set_property(TARGET MiniDNNLib PROPERTY CUDA_ARCHITECTURES 86)


