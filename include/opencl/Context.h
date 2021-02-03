/*
 * Context.h
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#ifndef OPENCL_INCLUDE_OPENCL_CONTEXT_H_
#define OPENCL_INCLUDE_OPENCL_CONTEXT_H_

#include <opencl/CommandQueue.h>

#include <CL/cl.h>

#include <vector>
#include <string>
#include <memory>
#include <mutex>


namespace opencl {

extern std::mutex g_mutex;

extern cl_platform_id g_platform;

extern cl_context g_context;


void create_context(const std::string& platform_name, cl_device_type device_type);

void release_context();

std::vector<cl_device_id> get_devices();

std::shared_ptr<CommandQueue> create_command_queue(int device);

std::string get_error_string(cl_int error);


} // opencl

#endif /* OPENCL_INCLUDE_OPENCL_CONTEXT_H_ */
