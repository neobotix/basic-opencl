/*
 * Context.cpp
 *
 *  Created on: Jul 12, 2018
 *      Author: dev
 */

#include <automy/basic_opencl/Context.h>

#include <mutex>


namespace automy {
namespace basic_opencl {

std::vector<cl_platform_id> get_platforms()
{
	cl_uint num_platforms = 0;
	std::vector<cl_platform_id> platforms(16);

	if(cl_int err = clGetPlatformIDs(platforms.size(), platforms.data(), &num_platforms)) {
		if(err != CL_PLATFORM_NOT_FOUND_KHR) {
			throw std::runtime_error("clGetPlatformIDs() failed with: " + get_error_string(err));
		}
	}
	platforms.resize(num_platforms);
	return platforms;
}

std::string get_platform_name(cl_platform_id platform)
{
	char name[1024] = {};
	if(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, 0)) {
		throw std::runtime_error("clGetPlatformInfo() failed");
	}
	return std::string(name);
}

cl_platform_id find_platform_by_name(const std::string& name)
{
	for(auto platform : get_platforms()) {
		if(get_platform_name(platform) == name) {
			return platform;
		}
	}
	return nullptr;
}

cl_context create_context(cl_platform_id platform, const std::vector<cl_device_id>& devices)
{
	cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
	cl_int err = 0;
	const auto context = clCreateContext(props, devices.size(), devices.data(), 0, 0, &err);
	if(err) {
		throw std::runtime_error("clCreateContext() failed with " + get_error_string(err));
	}
	return context;
}

void release_context(cl_context& context)
{
	if(context) {
		if(cl_int err = clReleaseContext(context)) {
			throw std::runtime_error("clReleaseContext() failed with " + get_error_string(err));
		}
		context = nullptr;
	}
}

std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type device_type)
{
	cl_uint num_devices = 0;
	std::vector<cl_device_id> device_list(256);
	if(cl_int err = clGetDeviceIDs(platform, device_type, device_list.size(), device_list.data(), &num_devices)) {
		device_list.clear();
		throw std::runtime_error("clGetDeviceIDs() failed with: " + get_error_string(err));
	}
	device_list.resize(num_devices);
	return device_list;
}

cl_device_id get_device(cl_platform_id platform, cl_device_type device_type, cl_uint device)
{
	const auto device_list = get_devices(platform, device_type);
	if(device >= device_list.size()) {
		throw std::runtime_error("no such OpenCL device: " + std::to_string(device));
	}
	return device_list[device];
}

std::string get_device_name(cl_device_id device_id)
{
	char dev_name[256] = {};
	size_t dev_name_len = 0;
	if(cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(dev_name), dev_name, &dev_name_len)) {
		throw std::runtime_error("clGetDeviceInfo() failed with " + get_error_string(err));
	}
	return std::string(dev_name, dev_name_len > 0 ? dev_name_len - 1 : 0);
}

std::shared_ptr<CommandQueue> create_command_queue(cl_context context, cl_device_id device)
{
	cl_int err = 0;
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
	if(err) {
		throw std::logic_error("clCreateCommandQueue() failed with " + get_error_string(err));
	}
	return CommandQueue::create(queue);
}

std::string get_error_string(cl_int error)
{
	switch(error){
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		
		// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
		
		// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
    }
}


} // basic_opencl
} // automy
