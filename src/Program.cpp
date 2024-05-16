/*
 * Program.cpp
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#include <automy/basic_opencl/Program.h>

#include <map>
#include <set>
#include <mutex>
#include <fstream>


namespace automy {
namespace basic_opencl {

std::shared_ptr<Program> Program::create(cl_context context) {
	return std::make_shared<Program>(context);
}

Program::Program(cl_context context)
	:	context(context)
{
}

Program::~Program() {
	if(program) {
		clReleaseProgram(program);
	}
}

void Program::add_include_path(const std::string& path) {
	includes.insert(path);
}

void Program::add_source(const std::string& file_name)
{
	std::vector<std::string> source_dirs {""};
	source_dirs.insert(source_dirs.end(), includes.begin(), includes.end());

	for(const auto& dir : source_dirs) {
		std::ifstream in(dir + file_name);
		if(in.good()) {
			sources.push_back(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
			return;
		}
	}
	throw std::runtime_error("no such file: '" + file_name + "'");
}

void Program::add_source_code(const std::string& source)
{
	sources.push_back(source);
}

void Program::add_binary(cl_device_id device, const std::string& file_name){
	std::vector<std::string> source_dirs {""};
	source_dirs.insert(source_dirs.end(), includes.begin(), includes.end());

	for(const auto& dir : source_dirs) {
		std::ifstream in(dir + file_name);
		if(in.good()) {
			binaries[device] = std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
			return;
		}
	}
	throw std::runtime_error("no such file: '" + file_name + "'");
}

void Program::add_binary_code(cl_device_id device, const std::string& binary){
	binaries[device] = binary;
}

void Program::create_from_source() {
	if(program) {
		throw std::logic_error("program already created");
	}
	
	std::vector<const char*> list;
	for(const std::string& source : sources) {
		list.push_back(source.c_str());
	}
	
	cl_int err = 0;
	program = clCreateProgramWithSource(context, list.size(), &list[0], 0, &err);
	if(err) {
		throw opencl_error_t("clCreateProgramWithSource() failed with " + get_error_string(err));
	}
}

void Program::create_from_binary() {
	if(program) {
		throw std::logic_error("program already created");
	}

	std::vector<cl_device_id> device_ids;
	std::vector<size_t> binary_lengths;
	std::vector<const unsigned char *> binary_code;
	for(const auto& entry : binaries) {
		device_ids.push_back(entry.first);
		binary_lengths.push_back(entry.second.size());
		binary_code.push_back(reinterpret_cast<const unsigned char *>(entry.second.c_str()));
	}
	std::vector<cl_int> binary_status(binary_code.size(), CL_SUCCESS);

	cl_int error = CL_SUCCESS;
	program = clCreateProgramWithBinary(context, device_ids.size(), device_ids.data(), binary_lengths.data(), binary_code.data(), binary_status.data(), &error);
	for(size_t i=0; i<device_ids.size(); i++) {
		const auto status = binary_status[i];
		if(status != CL_SUCCESS) {
			throw opencl_error_t("clCreateProgramWithBinary() failed at device " + std::to_string(i) + " with: " + get_error_string(status));
		}
	}
	if(error != CL_SUCCESS) {
		throw opencl_error_t("clCreateProgramWithBinary() failed with: " + get_error_string(error));
	}
}

bool Program::build(const std::vector<cl_device_id>& devices, bool with_arg_names)
{
	if(!program) {
		throw std::logic_error("program == nullptr");
	}
	have_arg_info = with_arg_names;
	
	std::string options_ = options;
	if(with_arg_names) {
		options_ += " -cl-kernel-arg-info";
	}
	for(const auto& path : includes) {
		if(!path.empty()) {
			options_ += " -I " + path;
		}
	}
	
	bool success = true;
	if(cl_int err = clBuildProgram(program, devices.size(), devices.data(), options_.c_str(), 0, 0)) {
		if(err != CL_BUILD_PROGRAM_FAILURE) {
			throw opencl_error_t("clBuildProgram() failed with " + get_error_string(err));
		}
		success = false;
	}
	
	for(cl_device_id device : devices) {
		size_t length = 0;
		cl_build_status status;
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, &length)) {
			throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_STATUS) failed with " + get_error_string(err));
		}
		if(status != CL_BUILD_SUCCESS) {
			success = false;
		}
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &length)) {
			throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG, 0, 0) failed with " + get_error_string(err));
		}
		if(length > 0) {
			std::string log;
			log.resize(length);
			if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), &log[0], &length)) {
				throw opencl_error_t("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG) failed with " + get_error_string(err));
			}
			if(length) {
				log.resize(length - 1);
			}
			build_log.push_back(log);
		}
	}
	return success;
}

std::vector<std::string> Program::get_sources() const {
	return sources;
}

std::map<cl_device_id, std::string> Program::get_binaries() const {
	if(!program) {
		throw std::logic_error("program == nullptr");
	}

	size_t num_devices = 0;
	{
		const auto error = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
		if(error != CL_SUCCESS) {
			throw opencl_error_t("clGetProgramInfo(CL_PROGRAM_NUM_DEVICES) failed with: " + get_error_string(error));
		}
	}
	std::vector<cl_device_id> device_ids(num_devices);
	{
		const auto error = clGetProgramInfo(program, CL_PROGRAM_DEVICES, device_ids.size()*sizeof(device_ids[0]), device_ids.data(), NULL);
		if(error != CL_SUCCESS) {
			throw opencl_error_t("clGetProgramInfo(CL_PROGRAM_DEVICES) failed with: " + get_error_string(error));
		}
	}
	std::vector<std::vector<unsigned char>> binaries(num_devices);
	{
		std::vector<size_t> binary_sizes(num_devices);
		const auto error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, binary_sizes.size()*sizeof(binary_sizes[0]), binary_sizes.data(), NULL);
		if(error != CL_SUCCESS) {
			throw opencl_error_t("clGetProgramInfo(CL_PROGRAM_BINARY_SIZES) failed with: " + get_error_string(error));
		}
		for(size_t i=0; i<num_devices; i++) {
			binaries[i].resize(binary_sizes[i]);
		}
	}
	{
		std::vector<unsigned char *> binaries_param;
		for(auto &binary : binaries) {
			binaries_param.push_back(binary.data());
		}
		const auto error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_param.size()*sizeof(binaries_param[0]), binaries_param.data(), NULL);
		if(error != CL_SUCCESS) {
			throw opencl_error_t("clGetProgramInfo(CL_PROGRAM_BINARIES) failed with: " + get_error_string(error));
		}
	}

	std::map<cl_device_id, std::string> result;
	for(size_t i=0; i<num_devices; i++) {
		result[device_ids[i]] = std::string(binaries[i].begin(), binaries[i].end());
	}
	return result;
}

void Program::print_sources(std::ostream& out) const {
	for(const std::string& source : sources) {
		out << source << std::endl;
	}
}

void Program::print_build_log(std::ostream& out) const {
	for(const std::string& log : build_log) {
		out << log << std::endl;
	}
}

std::shared_ptr<Kernel> Program::create_kernel(const std::string& name) const {
	cl_int err = 0;
	cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
	if(err) {
		throw opencl_error_t("clCreateKernel() failed for '" + name + "' with " + get_error_string(err));
	}
	return Kernel::create(kernel, have_arg_info);
}


} // basic_opencl
} // automy
