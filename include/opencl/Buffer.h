/*
 * Buffer.h
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#ifndef OPENCL_INCLUDE_BUFFER_H_
#define OPENCL_INCLUDE_BUFFER_H_

#include <opencl/Context.h>


namespace opencl {

class Buffer {
public:
	Buffer() {}
	
	~Buffer() {
		if(data_) {
			clReleaseMemObject(data_);
		}
	}
	
	Buffer(const Buffer& buf) = delete;
	Buffer& operator=(const Buffer& buf) = delete;
	
	cl_mem data() const {
		return data_;
	}
	
protected:
	cl_mem data_ = 0;
	
};


} // opencl


#endif /* OPENCL_INCLUDE_BUFFER_H_ */
