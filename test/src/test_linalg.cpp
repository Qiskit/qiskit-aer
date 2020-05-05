#include <map>

#include <controllers/qasm_controller.hpp>
#include <framework/linalg/linalg.hpp>
#include <framework/types.hpp>
#include <type_traits>

#include "utils.hpp"

#define CATCH_CONFIG_MAIN

#include <catch.hpp>

template<typename T>
matrix<std::complex<T>> create_psd_matrix(){
    auto psd_matrix = matrix<std::complex<T>>(3,3);

    psd_matrix(0,0) = std::complex<T>(2., 0.);
    psd_matrix(0,1) = std::complex<T>(2., 1.);
    psd_matrix(0,2) = std::complex<T>(4., 0.);
                                     
    psd_matrix(1,0) = std::complex<T>(2., -1.);
    psd_matrix(1,1) = std::complex<T>(3., 0.);
    psd_matrix(1,2) = std::complex<T>(0., 1.);
                                     
    psd_matrix(2,0) = std::complex<T>(4., 0.);
    psd_matrix(2,1) = std::complex<T>(0., -1.);
    psd_matrix(2,2) = std::complex<T>(1., 0.);
    return psd_matrix;
}

template<typename T>
matrix<std::complex<T>> create_expected_eigenvectors(){
    matrix<std::complex<T>> expected_eigenvectors(3, 3);

    expected_eigenvectors(0,0) = std::complex<T>(1.31961, 0.215478);
    expected_eigenvectors(0,1) = std::complex<T>(0.861912, 0.0336195);
    expected_eigenvectors(0,2) = std::complex<T>(1., 0.);

    expected_eigenvectors(1,0) = std::complex<T>(-0.955925, 0.0745055);
    expected_eigenvectors(1,1) = std::complex<T>(0.298022, 0.341426);
    expected_eigenvectors(1,2) = std::complex<T>(1., 0.);

    expected_eigenvectors(2,0) = std::complex<T>(0.309772, 0.390218);
    expected_eigenvectors(2,1) = std::complex<T>(-1.56087, 0.613993);
    expected_eigenvectors(2,2) = std::complex<T>(1.,0.);
    return expected_eigenvectors;
}

template<typename T>
std::vector<T> create_expected_eigenvalues(){
    return { 6.31205, -3.16513, 2.85308 };
}

TEST_CASE("Linear Algebra utilities", "[eigen_psd]") {

    SECTION("the input matrix of complex of doubles, is a PSD"){
        auto psd_matrix_double = create_psd_matrix<double>();

        auto expected_eigenvalues = create_expected_eigenvalues<double>();
        auto expected_eigenvectors = create_expected_eigenvectors<double>();
        std::vector<double> eigenvalues;
        matrix<std::complex<double>> eigenvectors;
        eigensystem_psd(psd_matrix_double, eigenvalues, eigenvectors);

        for(size_t i = 0; i < expected_eigenvalues.size(); ++i){
            REQUIRE(AER::Linalg::almost_equal(
                expected_eigenvalues[i], eigenvalues[i]
            ));
        }
        REQUIRE(AER::Linalg::almost_equal(expected_eigenvectors, eigenvectors));
        //for(size_t i = 0; i < expected_eigenvectors.size(); ++i){
        //    for(size_t j = 0; i < expected_eigenvectors.size(); ++j){
        //        REQUIRE(Linalg::almost_equal(
        //            expected_eigenvectors[i][j], eigenvectors[i][j]
        //        ));
        //    }
        //}
    }
}
/*
    SECTION("the input matrix of complex of floats, is a PSD"){
        auto psd_matrix_float = create_psd_matrix<complexf_t>();

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
}*/

TEST_CASE( "Framework Utilities", "[almost_equal]" ) {
    SECTION( "The maximum difference between two scalars over 1.0 is greater than epsilon, so they are amlmost equal" ) {
        double first = 1.0 + std::numeric_limits<double>::epsilon();
        double actual = 1.0;
        // Because the max_diff param is bigger than epsilon, this should be almost equal
        REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-15, 1e-15));
    }

    SECTION( "The difference between two scalars really close to 0 should say are almost equal" ) {
        double first = 5e-323; // Really close to the min magnitude of double
        double actual = 6e-323;
        REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-323, 1e-323));
    }

    SECTION( "The maximum difference between two complex of doubles over 1.0 is greater than epsilon, so they are amlmost equal" ) {
        std::complex<double> first = {
            1.0 + std::numeric_limits<double>::epsilon(),
            1.0 + std::numeric_limits<double>::epsilon()
        };
        std::complex<double> actual {1.0, 1.0};
        // Because the max_diff param is bigger than epsilon, this should be almost equal
        REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-15, 1e-15));
    }

    SECTION( "The difference between two complex of doubles really close to 0 should say are almost equal" ) {
        std::complex<double> first = {5e-323, 5e-323}; // Really close to the min magnitude of double
        std::complex<double> actual = {6e-323, 6e-323};

        REQUIRE(AER::Linalg::almost_equal(first, actual)); // 1e-323, 1e-323));
    }
}
