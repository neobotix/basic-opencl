cmake_minimum_required(VERSION 3.2)
project(automy-basic-opencl)
add_compile_options(-DNOGDI)

file(GLOB GENERATED_SOURCE_FILES generated/src/*.cpp)

add_library(automy_basic_opencl SHARED
	${GENERATED_SOURCE_FILES}
	src/Context.cpp
	src/Kernel.cpp
	src/Program.cpp
)

target_include_directories(automy_basic_opencl
	PUBLIC include
)

target_compile_definitions(automy_basic_opencl
	PUBLIC CL_TARGET_OPENCL_VERSION=120
)

target_link_libraries(automy_basic_opencl
	OpenCL
)

install(DIRECTORY kernel/ DESTINATION kernel)
install(DIRECTORY include/ DESTINATION include)

install(TARGETS automy_basic_opencl DESTINATION lib)
