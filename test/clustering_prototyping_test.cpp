#include <cstddef>
#include <cstdint>
#include <print>
#include <ranges>
#include <vector>

#include "acutest/acutest.h"
#include "algebra/algebra.hpp"
#include "data_objects/aliases.hpp"
#include "epsilon.hpp"
#include "gmm_em/gmm_em.hpp"

using namespace cf;

constexpr std::uint8_t NR_FEATURES = 2;

// TODO: port to doctest

static void test_e_step(void) {
    std::vector<double> data{1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 8.0, 8.0, 6.0, 9.0, 9.0};

    std::vector<double> responsibilities(static_cast<std::size_t>(6 * NR_FEATURES), 0);

    std::vector<double> mixing_coefficients_data{0.5, 0.5};

    std::vector<double> covariances_data{1.0, 0.5, 0.5, 1.0, 1.0, 0.2, 0.2, 1.0};

    std::vector<double> means_data{2.0, 2.0, 7.0, 7.0};

    auto means = Means<2, 2>(means_data.data());
    auto covariances = Covariances<2, 2>(covariances_data.data());
    auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data);

    auto data_v = Data<2>(data.data(), 6);
    auto responsibilities_v = Responsibilities<2>(responsibilities.data(), 6);

    gmm::e_step<2, 2>(data_v, responsibilities_v, means, covariances, mixing_coefficients);

    for (auto data_point_index : std::views::iota(0UZ, responsibilities_v.extent(0))) {
        for (auto component_index : std::views::iota(0UZ, responsibilities_v.extent(1))) {
            std::print("{} ", responsibilities_v[data_point_index, component_index]);
        }
        std::println("");
    }
}

void test_m_step(void) {
    std::vector<double> data{1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0};

    std::vector<double> responsibilities{0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9};

    std::vector<double> mixing_coefficients_data(2, 0);

    std::vector<double> covariances_data(static_cast<std::size_t>(2 * 2 * 2), 0);

    std::vector<double> means_data(static_cast<std::size_t>(2 * 2), 0);

    auto means = Means<2, 2>(means_data.data());
    auto covariances = Covariances<2, 2>(covariances_data.data());
    auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data);

    auto data_v = Data<2>(data.data(), 6);
    auto responsibilities_v = Responsibilities<2>(responsibilities.data(), 6);

    gmm::m_step<2, 2>(data_v, responsibilities_v, means, covariances, mixing_coefficients);
    gmm::pretty_print_result<2, 2>(means, covariances, mixing_coefficients);
}

void test_log_likelihood(void) {
    std::vector<double> data{1.0, 2.0, 2.0, 1.0, 3.0, 4.0};

    std::vector<double> responsibilities(static_cast<std::size_t>(3 * NR_FEATURES), 0);

    std::vector<double> mixing_coefficients_data{0.5, 0.5};

    std::vector<double> covariances_data{1.0, 0.5, 0.5, 1.0, 1.0, 0.2, 0.2, 1.0};

    std::vector<double> means_data{0.0, 0.0, 3.0, 3.0};

    auto means = Means<2, 2>(means_data.data());
    auto covariances = Covariances<2, 2>(covariances_data.data());
    auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data);

    auto data_v = Data<2>(data.data(), 3);
    auto responsibilities_v = Responsibilities<2>(responsibilities.data(), 3);

    auto ll = gmm::incomplete_log_likelihood<2, 2>(data_v, responsibilities_v, means, covariances, mixing_coefficients);

    std::println("{}", ll);
}

void test_pdf(void) {
    std::vector<double> data{1.0, 2.0};

    std::vector<double> covariances_data{1.0, 0.5, 0.5, 1.0};

    std::vector<double> means_data{0.0, 0.0};

    auto means = Means<1, 2>(means_data.data());
    auto covariances = Covariances<1, 2>(covariances_data.data());
    auto data_v = Data<2>(data.data(), 1);

    auto det_covariance = algebra::det<1, 2>(covariances, 0);
    auto inv_covariance_data = algebra::gauss_jordan<1, 2>(covariances, 0);

    auto inv_covariance = InvCovarianceMatrix<2>(inv_covariance_data.data());

    const double FACTOR = gmm::norm_factor(2, det_covariance);
    std::println("norm_factor : {}", FACTOR);

    auto pdf = gmm::gaussian_pdf<1, 2>(data_v, 0, means, 0, inv_covariance, FACTOR);

    std::println("{}", pdf);
    TEST_CHECK(pdf == 0.024871417406145686);
}

void TestInvMatrix(void) {
    std::vector<double> matrix_data = {4, 7, 2, 6};

    std::vector<double> expected_inverse = {0.6, -0.7, -0.2, 0.4};

    auto matrix = algebra::ArrayOfQuadraticMatrices<1, 2>(matrix_data.data());

    const auto RESULT = algebra::gauss_jordan<1, 2>(matrix, 0);

    TEST_CHECK(expected_inverse.size() == RESULT.size());
    for (size_t i = 0; i < expected_inverse.size(); ++i) {
        TEST_CHECK(equal_within_ulps(expected_inverse[i], RESULT[i], 1));
    }

    auto covariances_data =
        std::vector<double>{0.1826157040586288, -0.030168499854128984, -0.030168499854128984, 0.13740393745688573,
                            4.555599020215432,  0.8865183200782009,    0.8865183200782009,    0.2843275384146684,
                            0.5869940143861183, -1.8659741507706662,   -1.8659741507706662,   8.343758605566636,
                            2.949989700967138,  -0.748508303587861,    -0.748508303587861,    0.3466551411993789,
                            1.170091664282992,  0.1593929274088073,    0.1593929274088073,    0.14590030918947156};

    auto covariances = algebra::ArrayOfQuadraticMatrices<5, 2>(covariances_data.data());

    for (auto i = 0; i < 5; i++) {
        auto inv = algebra::gauss_jordan<5, 2>(covariances, i);
        for (auto x : inv) {
            std::println("{}", x);
        }
    }
}

void TestDetMatrix(void) {
    std::vector<double> mat1{1, 2, 3, 4};
    auto matrix1 = algebra::ArrayOfQuadraticMatrices<1, 2>(mat1.data());
    auto det_mat1 = algebra::det<1, 2>(matrix1, 0);

    TEST_CHECK(-2.0 == det_mat1);

    std::vector<double> mat2{6, 1, 1, 4, -2, 5, 2, 8, 7};
    auto matrix2 = algebra::ArrayOfQuadraticMatrices<1, 3>(mat2.data());
    auto det_mat2 = algebra::det<1, 3>(matrix2, 0);
    std::println("{}", det_mat2);

    TEST_CHECK(equal_within_ulps(-306.0, det_mat2, 1));
}

void TestComputeCovarianceMatrix(void) {
    auto data_data = std::vector<double>{2.5, 2.4, 0.5, 0.7, 2.2, 2.9, 1.9, 2.2, 3.1, 3.0,
                                         2.3, 2.7, 2.0, 1.6, 1.0, 1.1, 1.5, 1.6, 1.1, 0.9};
    auto data = Data<2>(data_data.data(), 10);

    auto centroids = std::vector<double>{1.81, 1.91};

    auto cov_size = 1 * 2 * 2;
    std::vector<double> covariances(cov_size, 0);
    auto cov_v = Covariances<1, 2>(covariances.data());

    auto labels = std::vector<std::uint8_t>(20, 0);
    auto points_in_cluster = std::vector<std::uint32_t>{9};

    for (auto x : std::views::iota(0UZ, cov_v.extent(1))) {
        for (auto y : std::views::iota(0UZ, cov_v.extent(2))) {
            for (auto data_point_index : std::views::iota(0UZ, data.extent(0))) {
                cov_v[labels.at(data_point_index), x, y] +=
                    (data[data_point_index, x] - centroids[x]) * (data[data_point_index, y] - centroids[y]);
            }

            for (auto component_index : std::views::iota(0UZ, cov_v.extent(0))) {
                cov_v[component_index, x, y] /= points_in_cluster[component_index];
            }
        }
    }

    double sum = 0;
    double prev = 0;
    auto index = 0;
    for (auto x : data_data) {
        if (index == 0) {
            prev = x - centroids[index];
            index = 1;
        } else if (index == 1) {
            sum += (x - centroids[index]) * prev;
            index = 0;
        }
    }
    std::println("{}", sum / (10 - 1));

    gmm::pretty_print_covariances<1, 2>(cov_v);
}

TEST_LIST = {
    {                "test_e_step",                 test_e_step},
    {                "test_m_step",                 test_m_step},
    {        "test_log_likelihood",         test_log_likelihood},
    {                   "test_pdf",                    test_pdf},
    {              "TestInvMatrix",               TestInvMatrix},
    {              "TestDetMatrix",               TestDetMatrix},
    {"TestComputeCovarianceMatrix", TestComputeCovarianceMatrix},
    {                         NULL,                        NULL},
};
