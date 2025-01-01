cmake_minimum_required(VERSION 3.26)
project(Overload_sr_examples
  VERSION 0.1.0
  DESCRIPTION "Code examples for senders/receivers for Overload."
  LANGUAGES CXX
)

find_package(Threads REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

set(STDEXEC_PATH CACHE PATH "Path to stdexec")
if (NOT STDEXEC_PATH)
  message(FATAL_ERROR "Please set STDEXEC_PATH to the path of stdexec")
endif()
if (NOT EXISTS "${STDEXEC_PATH}/include/stdexec/execution.hpp")
  message(FATAL_ERROR "${STDEXEC_PATH}/include/stdexec/execution.hpp does not exist")
endif()
message(STATUS "Using stdexec from ${STDEXEC_PATH}")

function(add_example name)
  add_executable(${name} "src/${name}.cpp")
  target_include_directories(${name} PUBLIC "${STDEXEC_PATH}/include" "${OpenCV_INCLUDE_DIRS}")
  target_compile_features(${name} PUBLIC cxx_std_20)
  target_compile_definitions(${name} PUBLIC STDEXEC_SYSTEM_CONTEXT_HEADER_ONLY=1)
  target_link_libraries(${name} Threads::Threads)
  target_link_libraries(${name} GLUT::GLUT OpenGL::GL)
  target_link_libraries(${name} "${OpenCV_LIBS}")
endfunction()

add_example(hello_concurrency)
add_example(mandelbrot)
add_example(sort)
add_example(imgproc)
