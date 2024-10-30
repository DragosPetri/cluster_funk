#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <print>
#include <ranges>
#include <vector>

#include "args.hxx"
#include "csvcpp/csvcpp.h"
#include "data_objects/aliases.hpp"
#include "gmm_em/gmm_em.hpp"
#include "kmeans/kmeans.hpp"

namespace fs = std::filesystem;

using namespace cf;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

constexpr std::uint8_t NR_COMPONENTS = 5;
constexpr std::uint8_t NR_FEATURES = 2;

static void
save_labeled_data(const Data<NR_FEATURES> &data, const std::vector<std::uint8_t> &labels, const std::string &output) {
    csv::CsvFile output_file;
    output_file.reserve(1 + data.extent(0));

    csv::CsvRow header;
    header.reserve(1 + NR_FEATURES + 1);
    header.emplace_back("Key");
    for (auto feature_index = 0; feature_index < NR_FEATURES; feature_index++) {
        header.emplace_back(std::format("Feature{}", feature_index + 1));
    }
    header.emplace_back("Cluster");

    output_file.emplace_back(header);

    for (auto data_point_index : std::views::iota(0uz, data.extent(0))) {
        csv::CsvRow row;
        row.reserve(NR_FEATURES + 2);

        row.emplace_back(static_cast<std::uint32_t>(data_point_index));

        for (auto feature_index : std::views::iota(0uz, data.extent(1))) {
            row.emplace_back(data[data_point_index, feature_index]);
        }

        row.emplace_back(labels.at(data_point_index));
        output_file.emplace_back(row);
    }

    output_file.save(output);
}

void save_cluster_centroids(const Means<NR_COMPONENTS, NR_FEATURES> &centroids, const std::string &output) {
    csv::CsvFile output_file;
    output_file.reserve(1 + centroids.extent(0));

    csv::CsvRow header;
    header.reserve(1 + NR_FEATURES + 1);
    for (auto feature_index = 0; feature_index < NR_FEATURES; feature_index++) {
        header.emplace_back(std::format("Feature{}", feature_index + 1));
    }

    output_file.emplace_back(header);

    for (auto cluster_index : std::views::iota(0uz, centroids.extent(0))) {
        csv::CsvRow row;
        row.reserve(centroids.extent(1));

        for (auto feature_index : std::views::iota(0uz, centroids.extent(1))) {
            row.emplace_back(centroids[cluster_index, feature_index]);
        }
        output_file.emplace_back(row);
    }

    output_file.save(output);
}

int main(int argc, char **argv) {
    args::ArgumentParser parser("This is a CLI for running Clustering algorithms", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Group arguments(parser, "arguments", args::Group::Validators::All, args::Options::Global);
    args::ValueFlag<std::string> dataset(arguments, "dataset_path", "Path to CSV dataset", {"dataset"});
    args::ValueFlag<std::string> output_path(arguments, "output_path", "Path to output labeled data to", {"output"});
    args::ValueFlag<std::size_t> max_iter(
        arguments, "max_iter", "Max Iteration the clustering algo should make", {"max_iter"}
    );
    args::ValueFlag<double> tolerance(arguments, "tolerance", "Tolerance of diff between iterations to quit", {"tol"});

    args::Group clustering_algorithms(parser, "Implemented clustering algorithms:", args::Group::Validators::Xor);
    args::Flag kmeans(clustering_algorithms, "kmeans", "Kmeans", {"kmeans"});
    args::Flag gmm_em(clustering_algorithms, "gmm_em", "Gaussian Mixture Model - Expectation Maximization", {"gmm"});
    args::Flag gmm_em_kmeans(
        clustering_algorithms,
        "gmm_em_kmeans",
        "Gaussian Mixture Model - Expectation Maximization with kmeans initialization",
        {"gmm_kmeans"}
    );
    args::Flag dbscan(clustering_algorithms, "dbscan", "DBSCAN", {"dbscan"});

    try {
        parser.ParseCLI(argc, argv);

        if (!fs::exists(args::get(dataset))) {
            std::println("Invalid Path!");
            return 0;
        }

        if (!fs::exists(args::get(output_path))) {
            fs::create_directories(fs::path(args::get(output_path)));
        }

        auto labeled_data_file_name = fs::path(args::get(output_path)) / fs::path(args::get(dataset)).filename();
        auto centroids_file_name = fs::path(args::get(output_path))
            / fs::path(std::format("centroids_{}", fs::path(args::get(dataset)).filename().string()));

        std::size_t max_iter_value = args::get(max_iter);
        double tolerance_value = args::get(tolerance);

        csv::CsvFile dataset_file{};

        try {
            std::println("Parsing CSV dataset with path {}", args::get(dataset));
            dataset_file.load(args::get(dataset));
            std::println("Parsed dataset");
        } catch (const std::exception &e) {
            std::println("Failed to parse CSV dataset with {}", e.what());
            return 0;
        }

        std::size_t data_rows = dataset_file.size();
        std::size_t data_cols = NR_FEATURES;

        std::vector<double> good_data;
        good_data.reserve(data_rows * data_cols);

        bool skip = true;
        for (std::size_t data_row = 0; data_row < data_rows; data_row++) {
            if (skip) {
                skip = false;
                continue;
            }
            for (std::size_t data_col = 1; data_col <= data_cols; data_col++) {
                good_data.emplace_back(dataset_file.at(data_row).at(data_col).as<double>());
            }
        }

        auto data_v = Data<NR_FEATURES>(good_data.data(), good_data.size() / NR_FEATURES);

        if (gmm_em) {
            auto t1 = high_resolution_clock::now();
            auto [means_data, covariances_data, mixing_coefficients_data] = gmm::fit_incomplete_log_likelihood<
                NR_COMPONENTS,
                NR_FEATURES>(data_v, max_iter_value, tolerance_value, gmm::random_init<NR_COMPONENTS, NR_FEATURES>);
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;

            std::println("Fit duration seconds {}", ms_double.count() / 1000);

            auto means = Means<NR_COMPONENTS, NR_FEATURES>(means_data.data());
            auto covariances = Covariances<NR_COMPONENTS, NR_FEATURES>(covariances_data.data());
            auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data.data(), NR_COMPONENTS);

            gmm::pretty_print_result<NR_COMPONENTS, NR_FEATURES>(means, covariances, mixing_coefficients);

            auto labels =
                gmm::assign_labels<NR_COMPONENTS, NR_FEATURES>(data_v, means, covariances, mixing_coefficients);

            save_labeled_data(data_v, labels, labeled_data_file_name.string());
            save_cluster_centroids(means, centroids_file_name);
        } else if (gmm_em_kmeans) {
            auto t1 = high_resolution_clock::now();
            auto [means_data, covariances_data, mixing_coefficients_data] = gmm::fit_incomplete_log_likelihood<
                NR_COMPONENTS,
                NR_FEATURES>(data_v, max_iter_value, tolerance_value, gmm::kmeans_init<NR_COMPONENTS, NR_FEATURES>);
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;

            std::println("Fit duration seconds {}", ms_double.count() / 1000);

            auto means = Means<NR_COMPONENTS, NR_FEATURES>(means_data.data());
            auto covariances = Covariances<NR_COMPONENTS, NR_FEATURES>(covariances_data.data());
            auto mixing_coefficients = MixingCoefficients(mixing_coefficients_data.data(), NR_COMPONENTS);

            gmm::pretty_print_result<NR_COMPONENTS, NR_FEATURES>(means, covariances, mixing_coefficients);

            auto labels =
                gmm::assign_labels<NR_COMPONENTS, NR_FEATURES>(data_v, means, covariances, mixing_coefficients);

            save_labeled_data(data_v, labels, labeled_data_file_name.string());
            save_cluster_centroids(means, centroids_file_name);
        } else if (kmeans) {
            auto t1 = high_resolution_clock::now();
            auto [centroids_data, labels] =
                kmeans::fit<NR_COMPONENTS, NR_FEATURES>(data_v, max_iter_value, tolerance_value);
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;

            std::println("Fit duration seconds {}", ms_double.count() / 1000);

            auto centroids = Means<NR_COMPONENTS, NR_FEATURES>(centroids_data.data());

            kmeans::pretty_print_result<NR_COMPONENTS, NR_FEATURES>(centroids);

            save_labeled_data(data_v, labels, labeled_data_file_name.string());
            save_cluster_centroids(centroids, centroids_file_name);
        } else {
            std::println("Not implemented yet");
            return 0;
        }
    } catch (const args::Help &) {
        std::println("{}", parser.Help());
        return 0;
    } catch (const args::ValidationError &) {
        std::println("{}", parser.Help());
        return 1;
    } catch (const args::ParseError &e) {
        std::println("{}", e.what());
        std::println("{}", parser.Help());
        return 1;
    }

    return 0;
}
