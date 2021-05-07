/*
 * Program.cpp
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#include <automy/basic_opencl/Program.h>

#include <map>
#include <fstream>


namespace automy {
namespace basic_opencl {

std::vector<std::string> g_includes;

std::map<std::string, std::shared_ptr<const Program>> g_programs;


void add_include_path(const std::string& path) {
	std::lock_guard<std::mutex> lock(g_mutex);
	g_includes.push_back(path);
}

std::shared_ptr<const Program> get_program(const std::string& name) {
	std::lock_guard<std::mutex> lock(g_mutex);
	auto it = g_programs.find(name);
	if(it != g_programs.end()) {
		return it->second;
	} else {
		throw std::runtime_error("get_program() undefined reference to program '" + name + "'");
	}
}

void register_program(const std::string& name, std::shared_ptr<const Program> program) {
	std::lock_guard<std::mutex> lock(g_mutex);
	g_programs[name] = program;
}


std::shared_ptr<Program> Program::create() {
	return std::make_shared<Program>();
}

Program::Program() {}

Program::~Program() {
	if(program) {
		clReleaseProgram(program);
	}
}

void Program::add_source(const std::string& file_name) {
	std::vector<std::string> source_dirs {""};
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		source_dirs.insert(source_dirs.end(), g_includes.begin(), g_includes.end());
	}
	for(const auto& dir : source_dirs) {
		std::ifstream in(dir + file_name);
		if(in.good()) {
			sources.push_back(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
			return;
		}
	}
	throw std::runtime_error("no such file: '" + file_name + "'");
}

void Program::create_from_source() {
	if(program) {
		clReleaseProgram(program);
		program = 0;
	}
	
	std::vector<const char*> list;
	for(const std::string& source : sources) {
		list.push_back(source.c_str());
	}
	
	cl_int err = 0;
	program = clCreateProgramWithSource(g_context, list.size(), &list[0], 0, &err);
	if(err) {
		throw std::runtime_error("clCreateProgramWithSource() failed with " + get_error_string(err));
	}
}

bool Program::build() {
	if(!program) {
		throw std::logic_error("program == 0");
	}
	std::lock_guard<std::mutex> lock(g_mutex);
	
	std::string options_ = options + " -cl-kernel-arg-info";
	for(const auto& path : g_includes) {
		if(!path.empty()) {
			options_ += " -I " + path;
		}
	}
	
	bool success = true;
	std::vector<cl_device_id> devices = get_devices();
	if(cl_int err = clBuildProgram(program, devices.size(), &devices[0], options_.c_str(), 0, 0)) {
		if(err != CL_BUILD_PROGRAM_FAILURE) {
			throw std::runtime_error("clBuildProgram() failed with " + get_error_string(err));
		}
		success = false;
	}
	
	for(cl_device_id device : devices) {
		size_t length = 0;
		cl_build_status status;
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, &length)) {
			throw std::runtime_error("clGetProgramBuildInfo(CL_PROGRAM_BUILD_STATUS) failed with " + get_error_string(err));
		}
		if(status != CL_BUILD_SUCCESS) {
			success = false;
		}
		if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 0, &length)) {
			throw std::runtime_error("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG, 0, 0) failed with " + get_error_string(err));
		}
		if(length > 2) {
			std::string log;
			log.resize(length - 1);
			if(cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size() + 1, &log[0], &length)) {
				throw std::runtime_error("clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG) failed with " + get_error_string(err));
			}
			build_log.push_back(log);
		}
	}
	return success;
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
		throw std::runtime_error("clCreateKernel() failed for '" + name + "' with " + get_error_string(err));
	}
	return Kernel::create(kernel);
}


} // basic_opencl
} // automy
