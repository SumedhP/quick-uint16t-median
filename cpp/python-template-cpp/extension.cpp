#include "python-template-cpp/extension.hpp"

#include <cmath>
#include <limits>
#include <vector>

float median_uint16_2d(py::array_t<uint16_t, py::array::c_style | py::array::forcecast> array, float non_zero_threshold) {
	const py::buffer_info info = array.request();
	if (info.ndim != 2) [[unlikely]] {
		throw std::runtime_error("median expects a 2D uint16 NumPy array");
	}

	if (non_zero_threshold < 0.0f || non_zero_threshold > 1.0f) [[unlikely]] {
		throw std::runtime_error("non_zero_threshold must be between 0 and 1");
	}

    // Calculate total number of elements and check for empty array
	const auto total = static_cast<std::size_t>(info.shape[0]) * static_cast<std::size_t>(info.shape[1]);
	if (total == 0) [[unlikely]] {
		throw std::runtime_error("median expects a non-empty array");
	}

	const auto *data = static_cast<const uint16_t *>(info.ptr);
	alignas(64) std::uint32_t counts[65536] = {};

	{
		py::gil_scoped_release release;
		for (std::size_t index = 0; index < total; ++index) {
			++counts[data[index]];
		}
	}

	const std::size_t zero_count = counts[0];
	const std::size_t non_zero_count = total - zero_count;
	if (static_cast<float>(non_zero_count) / static_cast<float>(total) < non_zero_threshold) [[unlikely]] {
		return std::numeric_limits<float>::quiet_NaN();
	}

	if (non_zero_count == 0) [[unlikely]] {
		return std::numeric_limits<float>::quiet_NaN();
	}

	const std::size_t target = (non_zero_count - 1) / 2;
	std::size_t seen = 0;
	for (std::size_t value = 1; value < 65536; ++value) {
		seen += counts[value];
		if (seen > target) {
			return static_cast<float>(value);
		}
	}

	throw std::runtime_error("median calculation failed");
}
