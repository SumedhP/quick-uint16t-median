#pragma once
#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

float median_uint16_2d(py::array_t<uint16_t, py::array::c_style | py::array::forcecast> array, float non_zero_threshold = 0.5);

PYBIND11_MODULE(extension, m) {
  m.doc() = "Fast uint16 median for 2D NumPy arrays";

  m.def("median", &median_uint16_2d, py::arg("array"), py::arg("non_zero_threshold") = 0.5,
        "Return the lower median of a 2D uint16 NumPy array, ignoring zeros.
         Returns NaN if the proportion of non-zero elements is below the specified threshold or if there are no non-zero elements.");
}
