cmake_minimum_required(VERSION 3.2)
project(automy-basic-opencl)

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

target_link_libraries(automy_basic_opencl
	automy_basic automy_math OpenCL
)

install(DIRECTORY kernel/ DESTINATION kernel)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY generated/include/ DESTINATION include)
install(DIRECTORY interface/ DESTINATION interface/automy/basic_opencl)

install(TARGETS automy_basic_opencl DESTINATION lib)
