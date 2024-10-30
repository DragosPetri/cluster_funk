#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <print>
#include <random>
#include <ranges>
#include <vector>

#include "data_objects/aliases.hpp"

namespace cf::kmeans {

    struct FitData {
        std::vector<double> centroids;
        std::vector<std::uint8_t> labels;
    };

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    FitData fit(Data<nr_features> data, std::size_t max_iter, double tolerance) {
        std::random_device random_device;
        std::mt19937 gen(random_device());
        std::uniform_int_distribution<std::uint32_t> distr(0, data.extent(0));

        std::vector<double> centroids_data(static_cast<std::size_t>(nr_features * nr_components), 0);
        auto centroids = Means<nr_components, nr_features>(centroids_data.data());

        for (auto component : std::views::iota(0UZ, centroids.extent(0))) {
            auto random_data_index = distr(gen);
            for (auto feature : std::views::iota(0UZ, centroids.extent(1))) {
                centroids[component, feature] = data[random_data_index, feature];
            }
        }

        std::vector<double> updated_centroids_data(static_cast<size_t>(nr_features * nr_components), 0);
        auto updated_centroids = Means<nr_components, nr_features>(updated_centroids_data.data());

        std::vector<std::uint8_t> labels(data.extent(0), -1);

        for (std::size_t i = 0; i < max_iter; i++) {
            std::array<std::uint32_t, nr_components> label_counter{};

            for (auto data_point_index : std::views::iota(0UZ, data.extent(0))) {
                double smallest_distance = DBL_MAX;
                std::uint8_t centroid = 0;

                for (auto centroid_index : std::views::iota(0UZ, centroids.extent(0))) {
                    double euclidean_distance = 0;

                    for (auto feature_index : std::views::iota(0UZ, centroids.extent(1))) {
                        euclidean_distance += std::pow(
                            data[data_point_index, feature_index] - centroids[centroid_index, feature_index], 2
                        );
                    }

                    euclidean_distance = std::sqrt(euclidean_distance);

                    if (euclidean_distance < smallest_distance) {
                        smallest_distance = euclidean_distance;
                        centroid = centroid_index;
                    }
                }

                for (auto feature_index : std::views::iota(0UZ, updated_centroids.extent(1))) {
                    updated_centroids[centroid, feature_index] += data[data_point_index, feature_index];
                }

                label_counter[centroid]++;
                labels[data_point_index] = centroid;
            }

            double norm = 0;
            for (auto centroid_index : std::views::iota(0UZ, centroids.extent(0))) {
                for (auto feature_index : std::views::iota(0UZ, centroids.extent(1))) {
                    updated_centroids[centroid_index, feature_index] /= label_counter[centroid_index];
                    norm += std::pow(
                        updated_centroids[centroid_index, feature_index] - centroids[centroid_index, feature_index], 2
                    );
                }
            }
            norm = std::sqrt(norm);

            if (norm < tolerance) {
                std::println("it: {}", i);
                break;
            }

            for (auto centroid_index : std::views::iota(0UZ, centroids.extent(0))) {
                for (auto feature_index : std::views::iota(0UZ, centroids.extent(1))) {
                    centroids[centroid_index, feature_index] = updated_centroids[centroid_index, feature_index];

                    updated_centroids[centroid_index, feature_index] = 0;
                }
            }
        }

        return {.centroids = centroids_data, .labels = labels};
    }

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    void pretty_print_result(const Means<nr_components, nr_features> centroids) {
        std::println("-------------------------------------------------------------------------------------");
        for (auto component : std::views::iota(0UZ, centroids.extent(0))) {
            std::print("Component {} : ", component + 1);
            for (auto feature : std::views::iota(0UZ, centroids.extent(1))) {
                std::print("{} ", centroids[component, feature]);
            }
            std::println("");
        }
        std::println("-------------------------------------------------------------------------------------");
    }

} // namespace cf::kmeans
