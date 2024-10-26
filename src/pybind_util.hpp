#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename T>
auto vector2pyarray(const std::vector<T>& vec) -> pybind11::array_t<T>
{
  return pybind11::array_t<T>(
    { vec.size() },
    { sizeof(T) },
    vec.data()
  );
}

template <typename T>
auto pyarray2vector(const pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>& pyarray) -> std::vector<T>
{
  const auto buffer_info = pyarray.request();
  T* ptr = static_cast<T*>(buffer_info.ptr);
  return std::vector<T>(ptr, ptr + buffer_info.shape[0]);
}