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
	Matrix() : Buffer3D<T>(Rows, Cols) {}

	Matrix(size_t depth) : Buffer3D<T>(Rows, Cols, depth) {}

	void resize(size_t depth) {
		Buffer3D<T>::resize(Rows, Cols, depth);
	}

	size_t rows() const {
		return Buffer3D<T>::width();
	}

	size_t cols() const {
		return Buffer3D<T>::height();
	}

	using Buffer3D<T>::depth;

	template<typename S>
	void upload(std::shared_ptr<CommandQueue> queue, const math::Matrix<S, Rows, Cols>& mat, bool blocking = false) {
		resize(1);
		math::Matrix<T, Rows, Cols> tmp(mat);
		upload(queue, tmp.get_data(), blocking);
	}

	void upload(std::shared_ptr<CommandQueue> queue, const std::vector<const math::Matrix<T, Rows, Cols>>& mats, bool blocking = false) {
		resize(mats.size());
		upload(queue, mats.data(), blocking);
	}

	void download(std::shared_ptr<CommandQueue> queue, math::Matrix<T, Rows, Cols>& mat, bool blocking = true) const {
		if(rows() != Rows  || cols() != Cols || depth() != 1) {
			throw std::logic_error("dimension mismatch");
		}
		download(queue, mat.get_data(), blocking);
	}

	void download(std::shared_ptr<CommandQueue> queue, const std::vector<const math::Matrix<T, Rows, Cols>>& mats, bool blocking = true) const {
		if(rows() != Rows  || cols() != Cols) {
			throw std::logic_error("dimension mismatch");
		}
		mats.resize(depth());
		download(queue, mats.data(), blocking);
	}

};


} // basic_opencl
} // automy

#endif /* INCLUDE_AUTOMY_BASIC_OPENCL_MATRIX_H_ */
