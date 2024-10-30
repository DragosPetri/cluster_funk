#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mdspan>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace cf::algebra {

    template<std::uint8_t nr_matrices, std::uint8_t size>
    using ArrayOfQuadraticMatrices = std::mdspan<double, std::extents<std::uint64_t, nr_matrices, size, size>>;

    // Inv matrix
    template<std::uint8_t nr_matrices, std::uint8_t size>
    std::vector<double>
    gauss_jordan(const ArrayOfQuadraticMatrices<nr_matrices, size> arr_of_matrices, std::uint8_t index_of_matrix) {
        std::vector<double> input_data{};
        input_data.reserve(arr_of_matrices.extent(1) * arr_of_matrices.extent(2));

        for (auto i : std::views::iota(0UZ, arr_of_matrices.extent(1))) {
            for (auto j : std::views::iota(0UZ, arr_of_matrices.extent(2))) {
                input_data.emplace_back(arr_of_matrices[index_of_matrix, i, j]);
            }
        }

        std::vector<double> inverse_data = input_data;

        auto input = std::mdspan(input_data.data(), arr_of_matrices.extent(1), arr_of_matrices.extent(2));
        auto inverse = std::mdspan(inverse_data.data(), arr_of_matrices.extent(1), arr_of_matrices.extent(2));

        for (auto i : std::views::iota(0UZ, inverse.extent(0))) {
            for (auto j : std::views::iota(0UZ, inverse.extent(1))) {
                inverse[i, j] = (i == j) ? 1 : 0;
            }
        }

        // Apply elementary row operations
        for (auto i : std::views::iota(0UZ, input.extent(0))) {
            // Check if the pivot element is zero
            if (std::abs(input[i, i]) == 0) {
                // Try to find a non-zero pivot element in the column
                for (std::size_t j = i + 1; j < input.extent(0); ++j) {
                    if (std::abs(input[j, i]) == 0) {
                        for (std::size_t col = 0; col < input.extent(0); ++col) {
                            std::swap(input[i, col], input[j, col]);
                        }
                        for (std::size_t col = 0; col < input.extent(0); ++col) {
                            std::swap(inverse[i, col], inverse[j, col]);
                        }
                        break;
                    }
                }
                // If no non-zero pivot found, matrix is singular
                return {};
            }

            // Make the pivot element 1
            double factor = 1 / input[i, i];

            for (std::size_t j = 0; j < input.extent(0); ++j) {
                input[i, j] *= factor;
                inverse[i, j] *= factor;
            }

            // Eliminate non-zero elements in the current column (except pivot row)
            for (std::size_t j = 0; j < input.extent(0); ++j) {
                if (i != j) {
                    double factor = input[j, i];
                    for (std::size_t k = 0; k < input.extent(0); ++k) {
                        input[j, k] -= factor * input[i, k];
                        inverse[j, k] -= factor * inverse[i, k];
                    }
                }
            }
        }

        return inverse_data;
    }

    template<std::uint8_t nr_matrices, std::uint8_t size>
    double det(const ArrayOfQuadraticMatrices<nr_matrices, size> arr_of_matrices, std::uint8_t index_of_matrix) {
        std::vector<double> input_data{};
        input_data.reserve(arr_of_matrices.extent(1) * arr_of_matrices.extent(2));

        for (auto i : std::views::iota(0UZ, arr_of_matrices.extent(1))) {
            for (auto j : std::views::iota(0UZ, arr_of_matrices.extent(2))) {
                input_data.emplace_back(arr_of_matrices[index_of_matrix, i, j]);
            }
        }

        auto input = std::mdspan(input_data.data(), arr_of_matrices.extent(1), arr_of_matrices.extent(2));

        if (size == 1) {
            return arr_of_matrices[index_of_matrix, 0, 0];
        }
        if (size == 2) {
            return (arr_of_matrices[index_of_matrix, 0, 0] * arr_of_matrices[index_of_matrix, 1, 1])
                - (arr_of_matrices[index_of_matrix, 0, 1] * arr_of_matrices[index_of_matrix, 1, 0]);
        }
        if (size == 3) {
            return (arr_of_matrices[index_of_matrix, 0, 0]
                    * (arr_of_matrices[index_of_matrix, 1, 1] * arr_of_matrices[index_of_matrix, 2, 2]
                       - arr_of_matrices[index_of_matrix, 1, 2] * arr_of_matrices[index_of_matrix, 2, 1]))
                - (arr_of_matrices[index_of_matrix, 0, 1]
                   * (arr_of_matrices[index_of_matrix, 1, 0] * arr_of_matrices[index_of_matrix, 2, 2]
                      - arr_of_matrices[index_of_matrix, 1, 2] * arr_of_matrices[index_of_matrix, 2, 0]))
                + (arr_of_matrices[index_of_matrix, 0, 2]
                   * (arr_of_matrices[index_of_matrix, 1, 0] * arr_of_matrices[index_of_matrix, 2, 1]
                      - arr_of_matrices[index_of_matrix, 1, 1] * arr_of_matrices[index_of_matrix, 2, 0]));
        }

        throw std::runtime_error("hahahahah");
    }

} // namespace cf::algebra
