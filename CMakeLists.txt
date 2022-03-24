cmake_minimum_required(VERSION 3.2)
project(automy-basic-opencl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(automy_basic_opencl SHARED
	src/Context.cpp
	src/Kernel.cpp
	src/Program.cpp
)

target_include_directories(automy_basic_opencl
	PUBLIC include
)

target_link_libraries(automy_basic_opencl
	OpenCL
)

target_compile_definitions(automy_basic_opencl PUBLIC NOGDI)

if(MSVC)
	include(GenerateExportHeader)
	GENERATE_EXPORT_HEADER(automy_basic_opencl)

	set_target_properties(automy_basic_opencl PROPERTIES ENABLE_EXPORTS ON)
	set_target_properties(automy_basic_opencl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

	find_library(OpenCL_LIBRARY OpenCL HINTS ${OpenCL_LIBRARIES})
	add_library(OpenCL INTERFACE IMPORTED)
	set_target_properties(OpenCL PROPERTIES INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}")

	target_include_directories(automy_basic_opencl
		PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${OpenCL_INCLUDE_DIR}
	)
endif()

install(DIRECTORY kernel/ DESTINATION kernel)
install(DIRECTORY include/ DESTINATION include)

install(TARGETS automy_basic_opencl DESTINATION lib)
