cmake_minimum_required(VERSION 3.10)
project(LlamaInteraction)

# 设置C++标准为C++11或更高版本
set(CMAKE_CXX_STANDARD 11)

# llama.cpp 项目的绝对路径
set(LLAMA_CPP_DIR "/home/team2/llama.cpp")

# 包含 llama.cpp 和 ggml 的头文件目录
include_directories(${LLAMA_CPP_DIR}/include)
include_directories(${LLAMA_CPP_DIR}/ggml/include)  # 使用 ggml.h 的路径

# 将 llama.cpp 源文件添加到你的项目中
add_library(llama ${LLAMA_CPP_DIR}/src/llama.cpp)

# 添加你的主源文件（llamaGo.cpp 的绝对路径）
add_executable(LlamaInteraction /home/team2/Desktop/kk/llamaGo.cpp)

# 链接 llama.cpp 库到你的项目
target_link_libraries(LlamaInteraction llama)

# 设置编译后的可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/team2/Desktop/kk/bin)
