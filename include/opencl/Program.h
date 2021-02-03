/*
 * Program.h
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#ifndef OPENCL_INCLUDE_OPENCL_PROGRAM_H_
#define OPENCL_INCLUDE_OPENCL_PROGRAM_H_

#include <opencl/Context.h>
#include <opencl/Kernel.h>

#include <vector>
#include <string>


namespace opencl {

class Program;

std::shared_ptr<const Program> get_program(const std::string& name);

void register_program(const std::string& name, std::shared_ptr<const Program> program);


class Program {
public:
	std::string options;
	
	std::vector<std::string> build_log;
	
	Program();
	
	Program(const Program&) = delete;
	Program& operator=(const Program&) = delete;
	
	~Program();
	
	static std::shared_ptr<Program> create();
	
	void add_source(const std::string& file_name);
	
	void create_from_source();
	
	bool build();
	
	void print_sources(std::ostream& out) const;
	
	void print_build_log(std::ostream& out) const;
	
	std::shared_ptr<Kernel> create_kernel(const std::string& name) const;
	
private:
	cl_program program = 0;
	
	std::vector<std::string> sources;
	
};


} // opencl


#endif /* OPENCL_INCLUDE_OPENCL_PROGRAM_H_ */
