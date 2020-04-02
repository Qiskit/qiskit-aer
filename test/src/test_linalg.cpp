#define CATCH_CONFIG_MAIN
#include <map>
#include <catch.hpp>

#include <controllers/qasm_controller.hpp>
#include <framework/linalg/linalg.hpp>
#include <framework/types.hpp>
#include <type_traits>

#include "utils.hpp"

namespace AER{
namespace Test{

template<typename matrix_inner_t>
matrix<matrix_inner_t> create_psd_matrix(){
    auto psd_matrix = matrix<matrix_inner_t>(3,3);

    psd_matrix(0,0) = matrix_inner_t(2., 0.);
    psd_matrix(0,1) = matrix_inner_t(2., 1.);
    psd_matrix(0,2) = matrix_inner_t(4., 0.);

    psd_matrix(1,0) = matrix_inner_t(2., -1.);
    psd_matrix(1,1) = matrix_inner_t(3., 0.);
    psd_matrix(1,2) = matrix_inner_t(0., 1.);

    psd_matrix(2,0) = matrix_inner_t(4., 0.);
    psd_matrix(2,1) = matrix_inner_t(0., -1.);
    psd_matrix(2,2) = matrix_inner_t(1., 0.);
    return psd_matrix;
}

template<typename floating_t>
std::vector<std::vector<std::complex<floating_t>>> create_expected_eigenvectors(){
    auto expected_egienvectors = std::vector<std::vector<std::complex<floating_t>>>{};
    using inner_complext_t = std::complex<floating_t>;

    expected_egienvectors[0][0] = inner_complext_t(1.31961, 0.215478);
    expected_egienvectors[0][1] = inner_complext_t(0.861912, 0.0336195);
    expected_egienvectors[0][2] = inner_complext_t(1., 0.);

    expected_egienvectors[1][0] = inner_complext_t(-0.955925, 0.0745055);
    expected_egienvectors[1][1] = inner_complext_t(0.298022, 0.341426);
    expected_egienvectors[1][2] = inner_complext_t(1., 0.);

    expected_egienvectors[2][0] = inner_complext_t(0.309772, 0.390218);
    expected_egienvectors[2][1] = inner_complext_t(-1.56087, 0.613993);
    expected_egienvectors[2][2] = inner_complext_t(1.,0.);
    return expected_egienvectors;
}

template<typename floating_t>
std::vector<std::complex<floating_t>> create_expected_eigenvalues(){
    using inner_complext_t = std::complex<floating_t>;
    return {
        inner_complex_t(6.31205,0),
        inner_complex_t(-3.16513,0),
        inner_complex_t(2.85308,0)
    };
}

TEST_CASE("Linear Algebra utilities", "[eigen_psd]") {
    auto psd_matrix_double = create_psd_matrix<complex_t>();
    auto psd_matrix_float = create_psd_matrix<complexf_t>();


    SECTION("the input matrix of complex of doubles, is a PSD"){
        auto expected_eigenvalues = create_expected_eigenvalues<double>();
        auto expected_eigenvectors = create_expected_eigenvectors<double>();
        std::vector<complex_t> eigenvalues;
        std::vector<std::vector<complex_t>> eigenvectors;
        eigensystem_psd(psd_matrix_double, eigenvalues, eigenvectors);

        for(size_t i = 0; i < expected_eigenvalues.size(); ++i){
            REQUIRE(Linalg::almost_equal(
                expected_eigenvalues[i], eigenvalues[i]
            ));
        }

        for(size_t i = 0; i < expected_eigenvectors.size(); ++i){
            for(size_t j = 0; i < expected_eigenvectors.size(); ++j){
                REQUIRE(Linalg::almost_equal(
                    expected_eigenvectors[i][j], eigenvectors[i][j]
                ));
            }
        }
    }

    SECTION("the input matrix of complex of floats, is a PSD"){
        auto expected_eigenvalues = create_expected_eigenvalues<float>();
        auto expected_eigenvectors = create_expected_eigenvectors<float>();
        std::vector<complexf_t> eigenvalues;
        std::vector<std::vector<complexf_t>> eigenvectors(3,3);
        eigensystem_psd(psd_matrix_float, eigenvalues, eigenvectors);

        for(size_t i = 0; i < expected_eigenvalues.size(); ++i){
            REQUIRE(Linalg::almost_equal(
                expected_eigenvalues[i], eigenvalues[i]
            ));
        }

        for(size_t i = 0; i < expected_eigenvectors.size(); ++i){
            for(size_t j = 0; i < expected_eigenvectors.size(); ++j){
                REQUIRE(Linalg::almost_equal(
                    expected_eigenvectors[i][j], eigenvectors[i][j]
                ));
            }
        }
    }

    SECTION("composing from the eigens should give us the original matrix"){
        std::vector<complex_t> eigenvalues;
        std::vector<std::vector<std::complex<double>>> eigenvectors;

        eigensystem_psd(psd_matrix_double, eigenvalues, eigenvectors);
        cmatrix_t expected_eigenvalues{psd_matrix_double.size()};

        for(auto j = 0; j < eigenvalues.size(); ++j){
            expected_eigenvalues +=
                eigenvalues[j] * Utils::projector(eigenvectors[j]);
        }

        for(size_t i = 0; i < expected_eigenvalues.size(); ++i){
            REQUIRE(Linalg::almost_equal(
                expected_eigenvalues[i], eigenvalues[i]
            ));
        }
    }
}

TEST_CASE( "Framework Utilities", "[almost_equal]" ) {
    SECTION( "The maximum difference between two scalars over 1.0 is greater than epsilon, so they are amlmost equal" ) {
        double first = 1.0 + std::numeric_limits<double>::epsilon();
        double actual = 1.0;
        // Because the max_diff param is bigger than epsilon, this should be almost equal
        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-15, 1e-15));
    }

    SECTION( "The difference between two scalars really close to 0 should say are almost equal" ) {
        double first = 5e-323; // Really close to the min magnitude of double
        double actual = 6e-323;
        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-323, 1e-323));
    }

    SECTION( "The maximum difference between two complex of doubles over 1.0 is greater than epsilon, so they are amlmost equal" ) {
        std::complex<double> first = {
            1.0 + std::numeric_limits<double>::epsilon(),
            1.0 + std::numeric_limits<double>::epsilon()
        };
        std::complex<double> actual = {1.0, 1.0};
        // Because the max_diff param is bigger than epsilon, this should be almost equal
        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-15, 1e-15));
    }

    SECTION( "The difference between two complex of doubles really close to 0 should say are almost equal" ) {
        std::complex<double> first = {5e-323, 5e-323}; // Really close to the min magnitude of double
        std::complex<double> actual = {6e-323, 6e-323};

        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-323, 1e-323));
    }
}

//------------------------------------------------------------------------------
} // end namespace Test
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
