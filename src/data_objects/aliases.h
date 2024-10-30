#pragma once
#include <mdspan>

namespace cf
{
    template<std::uint8_t nr_features>
    using Data = std::mdspan<double, std::extents<std::uint32_t, std::dynamic_extent, nr_features>>;

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    using Means = std::mdspan<double, std::extents<std::uint32_t, nr_components, nr_features>>;

    template<std::uint8_t nr_features>
    using InvCovarianceMatrix = std::mdspan<double, std::extents<std::uint64_t, nr_features, nr_features>>;

    template<std::uint8_t nr_components, std::uint8_t nr_features>
    using Covariances = std::mdspan<double, std::extents<std::uint32_t, nr_components, nr_features, nr_features>>;

    using MixingCoefficients = std::span<double>;

    template<std::uint8_t nr_components>
    using Responsibilities = std::mdspan<double, std::extents<std::uint64_t, std::dynamic_extent, nr_components>>;

} // namespace cf
