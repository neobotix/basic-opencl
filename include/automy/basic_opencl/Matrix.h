/*
 * Matrix.h
 *
 *  Created on: Feb 5, 2021
 *      Author: mad
 */

#ifndef INCLUDE_AUTOMY_BASIC_OPENCL_MATRIX_H_
#define INCLUDE_AUTOMY_BASIC_OPENCL_MATRIX_H_

#include <automy/basic_opencl/Buffer3D.h>

#include <automy/math/Matrix.hpp>


namespace automy {
namespace basic_opencl {

template<typename T, size_t Rows, size_t Cols>
class Matrix : public Buffer3D<T> {
public:
	Matrix() : Buffer3D<T>(){}

	Matrix(cl_context context) : Buffer3D<T>(context, Rows, Cols) {}

	Matrix(cl_context context, size_t depth) : Buffer3D<T>(context, Rows, Cols, depth) {}

	void resize(cl_context context, size_t depth) {
		Buffer3D<T>::resize(context, Rows, Cols, depth);
	}

	size_t rows() const {
		return Buffer3D<T>::width();
	}

	size_t cols() const {
		return Buffer3D<T>::height();
	}

	using Buffer3D<T>::depth;

#ifdef WITH_AUTOMY_MATH
	void upload(std::shared_ptr<CommandQueue> queue, const math::Matrix<T, Rows, Cols>& mat, bool blocking = true) {
		if(rows() != Rows || cols() != Cols || depth() != 1) {
			throw std::logic_error("dimension mismatch");
		}
		Buffer3D<T>::upload(queue, mat.get_data(), blocking);
	}

	template<typename S>
	void upload(std::shared_ptr<CommandQueue> queue, const math::Matrix<S, Rows, Cols>& mat, bool blocking = true) {
		const math::Matrix<T, Rows, Cols> tmp(mat);
		upload(queue, tmp, blocking);
	}

	void upload(std::shared_ptr<CommandQueue> queue, const std::vector<math::Matrix<T, Rows, Cols>>& mats, bool blocking = true) {
		if(depth() != mats.size()){
			throw std::logic_error("dimension mismatch");
		}
		Buffer3D<T>::upload(queue, mats[0].get_data(), blocking);
	}

	template<typename S>
	void upload(std::shared_ptr<CommandQueue> queue, const std::vector<S>& mats, bool blocking = true) {
		const std::vector<math::Matrix<T, Rows, Cols>> tmp(mats.begin(), mats.end());
		upload(queue, tmp, blocking);
	}

	void download(std::shared_ptr<CommandQueue> queue, math::Matrix<T, Rows, Cols>& mat, bool blocking = true) const {
		if(rows() != Rows  || cols() != Cols || depth() != 1) {
			throw std::logic_error("dimension mismatch");
		}
		Buffer3D<T>::download(queue, mat.get_data(), blocking);
	}

	void download(std::shared_ptr<CommandQueue> queue, std::vector<math::Matrix<T, Rows, Cols>>& mats, bool blocking = true) const {
		if(rows() != Rows  || cols() != Cols) {
			throw std::logic_error("dimension mismatch");
		}
		mats.resize(depth());
		Buffer3D<T>::download(queue, mats[0].get_data(), blocking);
	}
#endif

};


} // basic_opencl
} // automy

#endif /* INCLUDE_AUTOMY_BASIC_OPENCL_MATRIX_H_ */
