/*
 * Buffer2D.h
 *
 *  Created on: Jul 12, 2018
 *      Author: mad
 */

#ifndef OPENCL_INCLUDE_BUFFER2D_H_
#define OPENCL_INCLUDE_BUFFER2D_H_

#include <opencl/Buffer.h>

#include <automy/math/MatrixX.h>
#include <automy/basic/Image.h>


namespace opencl {

template<typename T>
class Buffer2D : public Buffer {
public:
	Buffer2D() {}
	
	Buffer2D(size_t width, size_t height) {
		resize(width, height);
	}
	
	static std::shared_ptr<Buffer2D<T>> create() {
		return std::make_shared<Buffer2D<T>>();
	}
	
	static std::shared_ptr<Buffer2D<T>> create(size_t width, size_t height) {
		return std::make_shared<Buffer2D<T>>(width, height);
	}
	
	void resize(size_t width, size_t height, size_t depth = 1) {
		if(width * height * depth != size()) {
			if(data_) {
				if(cl_int err = clReleaseMemObject(data_)) {
					throw std::runtime_error("clReleaseMemObject() failed with " + get_error_string(err));
				}
			}
			cl_int err = 0;
			data_ = clCreateBuffer(g_context, 0, width * height * depth * sizeof(T), 0, &err);
			if(err) {
				throw std::runtime_error("clCreateBuffer() failed with " + get_error_string(err));
			}
		}
		width_ = width;
		height_ = height;
		depth_ = depth;
	}
	
	size_t width() const {
		return width_;
	}
	
	size_t height() const {
		return height_;
	}
	
	size_t depth() const {
		return depth_;
	}
	
	size_t size() const {
		return width_ * height_ * depth_;
	}
	
	void upload(std::shared_ptr<CommandQueue> queue, const T* data, bool is_blocking = true) {
		if(cl_int err = clEnqueueWriteBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), data, 0, 0, 0)) {
			throw std::runtime_error("clEnqueueWriteBuffer() failed with " + get_error_string(err));
		}
	}
	
	void upload(std::shared_ptr<CommandQueue> queue, const std::vector<T>& vec, bool is_blocking = true) {
		resize(vec.size(), 1);
		if(cl_int err = clEnqueueWriteBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), vec.data(), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueWriteBuffer() failed with " + get_error_string(err));
		}
	}
	
	void upload(std::shared_ptr<CommandQueue> queue, const basic::Image<T>& img, bool is_blocking = true) {
		resize(img.width(), img.height(), img.depth());
		if(cl_int err = clEnqueueWriteBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), img.get_data(), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueWriteBuffer() failed with " + get_error_string(err));
		}
	}
	
	void upload(std::shared_ptr<CommandQueue> queue, const math::MatrixX<T>& mat, bool is_blocking = true) {
		resize(mat.rows(), mat.cols());
		if(cl_int err = clEnqueueWriteBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), mat.get_data(), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueWriteBuffer() failed with " + get_error_string(err));
		}
	}
	
	void download(std::shared_ptr<CommandQueue> queue, basic::Image<T>& img, bool is_blocking = true) const {
		img.resize(width_, height_, depth_);
		if(cl_int err = clEnqueueReadBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), img.get_data(), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueReadBuffer() failed with " + get_error_string(err));
		}
	}
	
	void download(std::shared_ptr<CommandQueue> queue, math::MatrixX<T>& mat, bool is_blocking = true) const {
		if(depth_ != 1) {
			throw std::logic_error("depth_ != 1");
		}
		mat.resize(width_, height_);
		if(cl_int err = clEnqueueReadBuffer(queue->get(), data_, is_blocking ? CL_TRUE : CL_FALSE, 0, size() * sizeof(T), mat.get_data(), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueReadBuffer() failed with " + get_error_string(err));
		}
	}
	
	void copy_from(std::shared_ptr<CommandQueue> queue, const Buffer2D<T>& other) {
		resize(other.width(), other.height(), other.depth());
		if(cl_int err = clEnqueueCopyBuffer(queue->get(), other.data(), data_, 0, 0, size() * sizeof(T), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueCopyBuffer() failed with " + get_error_string(err));
		}
	}
	
	void set_zero(std::shared_ptr<CommandQueue> queue) {
		const T zero = {};
		if(cl_int err = clEnqueueFillBuffer(queue->get(), data_, &zero, sizeof(T), 0, size() * sizeof(T), 0, 0, 0)) {
			throw std::runtime_error("clEnqueueFillBuffer() failed with " + get_error_string(err));
		}
	}
	
private:
	size_t width_ = 0;
	size_t height_ = 0;
	size_t depth_ = 0;
	
};


} // opencl


#endif /* OPENCL_INCLUDE_BUFFER2D_H_ */
