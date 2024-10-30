#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mdspan>
#include <numbers>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include "algebra/algebra.h"
#include "data_objects/aliases.h"
#include "kmeans/kmeans.h"

namespace cf::gmm {

    struct FittedParameters {
        std::vector<double> means;
        std::vector<double> covariances;
        std::vector<double> mixing_coefficients;
    };

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    struct InitializedParameters {
        std::vector<double> means;
        std::vector<double> covariances;
        std::vector<double> mixing_coefficients;
        std::vector<double> responsibilities;
    };


    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void pretty_print_means(Means<nr_components, nr_features> means) {
        // mu
        std::println("-------------------------------------------------------------------------------------");
        std::println("Means:");
        for (auto i : std::views::iota(0UZ, means.extent(0))) {
            for (auto j : std::views::iota(0UZ, means.extent(1))) {
                std::print("{} ", means[i, j]);
            }

            std::println("");
        }
        std::println("-------------------------------------------------------------------------------------");
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void pretty_print_covariances(Covariances<nr_components, nr_features> covariances) {
        // sigma
        std::println("-------------------------------------------------------------------------------------");
        std::println("Covariances:\n");
        for (auto component_index : std::views::iota(0UZ, covariances.extent(0))) {
            std::println("Component {} :\n", component_index + 1);
            for (auto i : std::views::iota(0UZ, covariances.extent(1))) {
                for (auto j : std::views::iota(0UZ, covariances.extent(2))) {
                    std::print("{} ", covariances[component_index, i, j]);
                }
                std::println("");
            }
            std::println("");
        }
        std::println("-------------------------------------------------------------------------------------");
    }

    inline void pretty_print_mixing_coefficients(MixingCoefficients mixing_coefficients) {
        // Pi
        std::println("-------------------------------------------------------------------------------------");
        std::println("Mixing coefficients:");
        for (auto i : std::views::iota(0UZ, mixing_coefficients.size())) {
            std::println("Component {} : {}", i, mixing_coefficients[i]);
        }
        std::println("");
        std::println("-------------------------------------------------------------------------------------");
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void pretty_print_result(
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        pretty_print_means<nr_components, nr_features>(means);
        pretty_print_covariances<nr_components, nr_features>(covariances);
        pretty_print_mixing_coefficients(mixing_coefficients);
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    InitializedParameters<nr_components, nr_features> random_init(const Data<nr_features> data) {
        auto data_size = data.extent(0);

        std::vector<double> mixing_coefficients;
        mixing_coefficients.reserve(nr_components);

        for (std::uint8_t i = 0; i < nr_components; i++) {
            mixing_coefficients.emplace_back(1.0 / nr_components);
        }

        std::vector<double> covariances(static_cast<std::size_t>(nr_components * nr_features * nr_features), 0);
        auto cov_v = Covariances<nr_components, nr_features>(covariances.data());

        for (auto i : std::views::iota(0UZ, cov_v.extent(0))) {
            for (auto j : std::views::iota(0UZ, cov_v.extent(1))) {
                cov_v[i, j, j] = 1;
            }
        }

        std::random_device random_device;
        std::mt19937 gen(random_device());
        std::normal_distribution<double> distribution(0, 1);

        std::vector<double> means;
        means.reserve(static_cast<std::size_t>(nr_components * nr_features));

        for (std::uint32_t i = 0; i < nr_components * 1 * nr_features; i++) {
            means.emplace_back(distribution(gen));
        }

        std::vector<double> responsibilities_data(data_size * nr_components, 0);

        return {means, covariances, mixing_coefficients, responsibilities_data};
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    InitializedParameters<nr_components, nr_features> kmeans_init(const Data<nr_features> data) {
        auto [centroids, labels] = kmeans::fit<nr_components, nr_features>(data, 100, 1e-6);
        std::println("{} size labels", labels.size());

        std::vector<double> mixing_coefficients(nr_components, 0);
        std::vector<std::uint64_t> points_in_cluster(nr_components, 0);

        for (auto label : labels) {
            ++points_in_cluster[label];
        }

        for (auto component_index : std::views::iota(0UZ, nr_components)) {
            mixing_coefficients[component_index] =
                static_cast<double>(points_in_cluster[component_index]) / static_cast<double>(data.extent(0));
        }

        auto cov_size = nr_components * nr_features * nr_features;
        std::vector<double> covariances(cov_size, 0);
        auto cov_v = Covariances<nr_components, nr_features>(covariances.data());
        auto means_v = Means<nr_components, nr_features>(centroids.data());

        pretty_print_means<nr_components, nr_features>(means_v);

        for (auto x : std::views::iota(0UZ, nr_features)) {
            for (auto y : std::views::iota(0UZ, nr_features)) {
                for (auto data_point_index : std::views::iota(0UZ, data.extent(0))) {
                    cov_v[labels.at(data_point_index), x, y] +=
                        (data[data_point_index, x] - means_v[labels.at(data_point_index), x])
                        * (data[data_point_index, y] - means_v[labels.at(data_point_index), x]);
                }

                for (auto component_index : std::views::iota(0UZ, nr_components)) {
                    cov_v[component_index, x, y] /= points_in_cluster[component_index] - 1;
                }
            }
        }

        std::vector<double> responsibilities_data(data.extent(0) * nr_components, 0);

        pretty_print_covariances<nr_components, nr_features>(cov_v);
        pretty_print_mixing_coefficients(MixingCoefficients(mixing_coefficients));

        return {
            .means = centroids,
            .covariances = covariances,
            .mixing_coefficients = mixing_coefficients,
            .responsibilities = responsibilities_data
        };
    }

    constexpr double norm_factor(std::uint8_t nr_features, double covariance_matrix_det) {
        return 1.0 / (std::sqrt((std::pow(2.0 * std::numbers::pi, nr_features))) * std::sqrt(covariance_matrix_det));
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    double gaussian_pdf(
        Data<nr_features> data,
        std::uint64_t index_of_data,
        Means<nr_components, nr_features> means,
        std::uint64_t index_of_means,
        InvCovarianceMatrix<nr_features> inv_covariance,
        double factor
    ) {
        double result = 0;

        for (auto inv_cov_row : std::views::iota(0UZ, inv_covariance.extent(0))) {
            double reduced_dot = 0;
            for (auto inv_cov_col : std::views::iota(0UZ, inv_covariance.extent(1))) {
                reduced_dot += (data[index_of_data, inv_cov_col] - means[index_of_means, inv_cov_col])
                    * inv_covariance[inv_cov_row, inv_cov_col];
            }
            result += (reduced_dot * (data[index_of_data, inv_cov_row] - means[index_of_means, inv_cov_row]));
        }

        result *= -0.5;
        result = std::exp(result);
        result *= factor;

        return result;
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void e_step(
        const Data<nr_features> data,
        const Responsibilities<nr_components> responsibilities,
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
            auto det_covariance = algebra::det<nr_components, nr_features>(covariances, component_index);
            auto inv_covariance_data = algebra::gauss_jordan<nr_components, nr_features>(covariances, component_index);

            auto inv_covariance = InvCovarianceMatrix<nr_features>(inv_covariance_data.data());

            const double FACTOR = norm_factor(nr_components, det_covariance);

            for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
                auto row_gaussian = gaussian_pdf<nr_components, nr_features>(
                    data, data_point_index, means, component_index, inv_covariance, FACTOR
                );

                responsibilities[data_point_index, component_index] =
                    mixing_coefficients[component_index] * row_gaussian;
            }
        }

        for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
            double summed_row = 0;
            for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
                summed_row += responsibilities[data_point_index, component_index];
            }

            for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
                responsibilities[data_point_index, component_index] /= summed_row;
            }
        }
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void m_step(
        const Data<nr_features> data,
        const Responsibilities<nr_components> responsibilities,
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        std::vector<double> summed_col_responsibilities(nr_components, 0.0);

        for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
            for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
                summed_col_responsibilities[component_index] += responsibilities[data_point_index, component_index];
            }
        }

        for (auto component_index : std::views::iota(0UZ, mixing_coefficients.size())) {
            mixing_coefficients[component_index] = summed_col_responsibilities[component_index] / data.extent(0);
        }

        // TODO : Consider calculating in favor of rows, might be more cache friendly
        for (auto component_index : std::views::iota(0UZ, means.extent(0))) {
            for (auto feature_index : std::views::iota(0UZ, means.extent(1))) {
                double new_value = 0;

                for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
                    new_value +=
                        responsibilities[data_point_index, component_index] * data[data_point_index, feature_index];
                }

                means[component_index, feature_index] = new_value / summed_col_responsibilities[component_index];
            }
        }

        for (auto component_index : std::views::iota(0UZ, covariances.extent(0))) {
            for (auto row_index : std::views::iota(0UZ, covariances.extent(1))) {
                for (auto col_index : std::views::iota(0UZ, covariances.extent(2))) {
                    double new_value = 0.0;

                    for (auto data_point_index : std::views::iota(0UZ, data.extent(0))) {
                        auto deviation_col = (data[data_point_index, col_index] - means[component_index, col_index]);

                        auto deviation_row = (data[data_point_index, row_index] - means[component_index, row_index]);

                        new_value +=
                            responsibilities[data_point_index, component_index] * deviation_col * deviation_row;
                    }

                    covariances[component_index, row_index, col_index] =
                        new_value / summed_col_responsibilities[component_index];
                }
            }
        }
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    double incomplete_log_likelihood(
        const Data<nr_features> data,
        const Responsibilities<nr_components> responsibilities,
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
            auto det_covariance = algebra::det<nr_components, nr_features>(covariances, component_index);
            auto inv_covariance_data = algebra::gauss_jordan<nr_components, nr_features>(covariances, component_index);

            auto inv_covariance = InvCovarianceMatrix<nr_features>(inv_covariance_data.data());

            const double FACTOR = norm_factor(nr_features, det_covariance);

            for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
                auto row_gaussian = gaussian_pdf<nr_components, nr_features>(
                    data, data_point_index, means, component_index, inv_covariance, FACTOR
                );

                responsibilities[data_point_index, component_index] =
                    mixing_coefficients[component_index] * row_gaussian;
            }
        }

        double log_likelihood = 0;

        for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
            double summed_row = 0;
            for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
                summed_row += responsibilities[data_point_index, component_index];
            }

            log_likelihood += std::log(summed_row);
        }

        return log_likelihood;
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    double complete_log_likelihood(
        const Data<nr_features> data,
        const Responsibilities<nr_components> responsibilities,
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
            auto det_covariance = algebra::det<nr_components, nr_features>(covariances, component_index);
            auto inv_covariance_data = algebra::gauss_jordan<nr_components, nr_features>(covariances, component_index);

            auto inv_covariance = inv_covariance_matrix<nr_features>(inv_covariance_data.data());

            const double FACTOR = norm_factor(nr_features, det_covariance);

            for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
                auto row_gaussian = gaussian_pdf<nr_components, nr_features>(
                    data, data_point_index, means, component_index, inv_covariance, FACTOR
                );

                responsibilities[data_point_index, component_index] =
                    responsibilities[data_point_index, component_index]
                    * std::log(
                        (mixing_coefficients[component_index] * row_gaussian)
                        / responsibilities[data_point_index, component_index]
                    );
            }
        }

        double log_likelihood = 0;

        for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
            for (auto component_index : std::views::iota(0UZ, responsibilities.extent(1))) {
                log_likelihood += responsibilities[data_point_index, component_index];
            }
        }

        return log_likelihood;
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    bool centroids_in_tolerance(
        const Means<nr_components, nr_features> old_centroids,
        const Means<nr_components, nr_features> new_centroids,
        double tolerance
    ) {
        for (auto cluster_index : std::views::iota(0UZ, old_centroids.extent(0))) {
            double distance = 0;

            for (auto feature_index : std::views::iota(0UZ, old_centroids.extent(1))) {
                distance += (new_centroids[cluster_index, feature_index] - old_centroids[cluster_index, feature_index])
                    * (new_centroids[cluster_index, feature_index] - old_centroids[cluster_index, feature_index]);
            }

            distance = std::sqrt(distance);

            if (distance > tolerance) {
                return false;
            }
        }

        return true;
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    FittedParameters fit_incomplete_log_likelihood(
        const Data<nr_features> data,
        std::size_t max_iter,
        double tolerance,
        std::function<InitializedParameters<nr_components, nr_features>(Data<nr_features>)> init_func
    ) {
        auto [means_data, covariances_data, mixing_coefficients_data, responsibilities_data] = init_func(data);

        auto means = Means<nr_components, nr_features>(means_data.data());
        auto covariances = Covariances<nr_components, nr_features>(covariances_data.data());
        auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data.data(), nr_components);

        auto responsibilities = Responsibilities<nr_components>(responsibilities_data.data(), data.extent(0));

        double prev_log_likelihood = -DBL_MAX;

        for (std::size_t i = 0; i < max_iter; i++) {
            e_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            m_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            const auto CURRENT_LOG_LIKELIHOOD = incomplete_log_likelihood<nr_components, nr_features>(
                data, responsibilities, means, covariances, mixing_coefficients
            );

            std::println("ll {}", CURRENT_LOG_LIKELIHOOD);

            if (std::fabs(CURRENT_LOG_LIKELIHOOD - prev_log_likelihood) < tolerance) {
                std::println("Converged at {}!", i);
                break;
            }

            prev_log_likelihood = CURRENT_LOG_LIKELIHOOD;
        }

        return {
            .means = std::move(means_data),
            .covariances = std::move(covariances_data),
            .mixing_coefficients = std::move(mixing_coefficients_data)
        };
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    FittedParameters fit_complete_log_likelihood(
        const Data<nr_features> data,
        std::size_t max_iter,
        double tolerance,
        std::function<InitializedParameters<nr_components, nr_features>(Data<nr_features>)> init_func
    ) {
        auto [means_data, covariances_data, mixing_coefficients_data, responsibilities_data] = init_func(data);

        auto means = Means<nr_components, nr_features>(means_data.data());
        auto covariances = Covariances<nr_components, nr_features>(covariances_data.data());
        auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data.data(), nr_components);

        auto responsibilities = Responsibilities<nr_components>(responsibilities_data.data(), data.extent(0));

        double prev_log_likelihood = -DBL_MAX;

        for (std::size_t i = 0; i < max_iter; i++) {
            e_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            m_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            const auto CURRENT_LOG_LIKELIHOOD = complete_log_likelihood<nr_components, nr_features>(
                data, responsibilities, means, covariances, mixing_coefficients
            );

            std::println("ll {}", CURRENT_LOG_LIKELIHOOD);

            if (std::fabs(CURRENT_LOG_LIKELIHOOD - prev_log_likelihood) < tolerance) {
                std::println("Converged at {}!", i);
                break;
            }

            prev_log_likelihood = CURRENT_LOG_LIKELIHOOD;
        }

        return {
            .means = std::move(means_data),
            .covariances = std::move(covariances_data),
            .mixing_coefficients = std::move(mixing_coefficients_data)
        };
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    FittedParameters fit_centroids(
        const Data<nr_features> data,
        std::size_t max_iter,
        double tolerance,
        std::function<InitializedParameters<nr_components, nr_features>(Data<nr_features>)> init_func
    ) {
        auto [means_data, covariances_data, mixing_coefficients_data, responsibilities_data] = init_func(data);

        auto means = Means<nr_components, nr_features>(means_data.data());
        auto covariances = Covariances<nr_components, nr_features>(covariances_data.data());
        auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data.data(), nr_components);

        auto responsibilities = Responsibilities<nr_components>(responsibilities_data.data(), data.extent(0));

        std::vector<double> old_centroids_data(means.extent(0) * means.extent(1), DBL_MAX);
        auto old_centroids = Means<nr_components, nr_features>(old_centroids_data.data());

        for (std::size_t i = 0; i < max_iter; i++) {
            e_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            m_step<nr_components, nr_features>(data, responsibilities, means, covariances, mixing_coefficients);

            if (centroids_in_tolerance<nr_components, nr_features>(old_centroids, means, tolerance)) {
                std::println("Converged at {}!", i);
                break;
            }

            old_centroids_data = means_data;
        }

        return {
            .means = std::move(means_data),
            .covariances = std::move(covariances_data),
            .mixing_coefficients = std::move(mixing_coefficients_data)
        };
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    std::vector<std::uint8_t> assign_labels(
        const Data<nr_features> data,
        const Means<nr_components, nr_features> means,
        const Covariances<nr_components, nr_features> covariances,
        const MixingCoefficients mixing_coefficients
    ) {
        std::vector<double> responsibilities_data(data.extent(0) * covariances.extent(0), 0);
        auto responsibilities = Responsibilities<nr_components>(responsibilities_data.data(), data.extent(0));

        for (std::uint8_t component = 0; component < nr_components; component++) {
            auto det_covariance = algebra::det<nr_components, nr_features>(covariances, component);
            auto inv_covariance_data = algebra::gauss_jordan<nr_components, nr_features>(covariances, component);

            auto inv_covariance = InvCovarianceMatrix<nr_features>(inv_covariance_data.data());

            const double FACTOR = norm_factor(nr_features, det_covariance);

            for (auto i : std::views::iota(0UZ, responsibilities.extent(0))) {
                auto row_gaussian =
                    gaussian_pdf<nr_components, nr_features>(data, i, means, component, inv_covariance, FACTOR);
                responsibilities[i, component] = mixing_coefficients[component] * row_gaussian;
            }
        }

        for (auto i : std::views::iota(0UZ, responsibilities.extent(0))) {
            double summed_row = 0;
            for (auto j : std::views::iota(0UZ, responsibilities.extent(1))) {
                summed_row += responsibilities[i, j];
            }

            for (auto j : std::views::iota(0UZ, responsibilities.extent(1))) {
                responsibilities[i, j] /= summed_row;
            }
        }

        std::vector<std::uint8_t> labels(data.extent(0), 99);

        for (auto data_point_index : std::views::iota(0UZ, responsibilities.extent(0))) {
            double smallest_distance = DBL_MIN;
            std::uint8_t nearest_cluster = -1;

            for (auto cluster_index : std::views::iota(0UZ, responsibilities.extent(1))) {
                if (responsibilities[data_point_index, cluster_index] > smallest_distance) {
                    smallest_distance = responsibilities[data_point_index, cluster_index];
                    nearest_cluster = cluster_index;
                }
            }

            labels.at(data_point_index) = nearest_cluster;
        }

        return labels;
    }

} // namespace cf::gmm
