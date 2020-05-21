#include <map>

#include <controllers/qasm_controller.hpp>
#include <framework/linalg/linalg.hpp>
#include <framework/types.hpp>
#include <type_traits>

#include "utils.hpp"

static std::random_device rd;
static std::default_random_engine rng(rd());

#define CATCH_CONFIG_MAIN

#include <catch.hpp>

template <typename T>
T rand_f(T min = 0.0, T max = 1.0) {
    std::uniform_real_distribution<T> distr(min, max) ;
    return distr(rng);
}

template<typename T>
std::complex<T> rand_z(T min = 0.0, T max = 1.0) {
    std::uniform_real_distribution<T> distr(min, max) ;
    return { distr(rng), distr(rng) };
}

template<typename T>
matrix<std::complex<T>> rand_hermitian_mat(size_t n, T range) {
    // we construct a random N * a matrix
    matrix<std::complex<T>> mat(n, n);
    for(size_t r=0; r<n; r++) {
        for(size_t c=r; c<n; c++) {
            if (r == c) { 
                mat(r, c) = std::complex<T>(
                                rand_f<T>(-1.0 * std::abs(range), std::abs(range)),
                                0.0); 
            } else {
                auto random_z = rand_z<T>(-1.0 * std::abs(range), std::abs(range));
                mat(r, c) = random_z;
                mat(c, r) = std::complex<T>(random_z.real(), -1 * random_z.imag());
            }
        }
    }
    return mat; //* AER::Utils::dagger(mat);
}

template<typename T>
matrix<std::complex<T>> herm_mat_2d_1() {
    auto mat = matrix<std::complex<T>>(2,2);
    mat(0,0) = std::complex<T>(1.0,0.0);
    mat(0,1) = std::complex<T>(1.0, -2.0);
    mat(1,0) = std::complex<T>(1.0, 2.0);
    mat(1,1) = std::complex<T>(2.0, 0.0);
    return mat;
}

template<typename T>
matrix<std::complex<T>> herm_mat_2d_eigenvectors_1() {
    auto mat = matrix<std::complex<T>>(2,2);
    mat(0,0) = std::complex<T>(-(std::sqrt(21.0)/10.0) - (1.0/10.0), (1.0/5.0) + (std::sqrt(21.0)/5.0));
    mat(1,0) = std::complex<T>(1.0, 0.0);
    mat(0,1) = std::complex<T>((std::sqrt(21.0)/10.0)-(1.0/10.0), (1.0/5.0) - (std::sqrt(21.0)/5.0));
    mat(1,1) = std::complex<T>(1.0, 0.0);
    return mat;
}

template<typename T>
std::vector<T> herm_mat_2d_eigenvalues_1(){
    return { 3.0/2.0 - std::sqrt(21.0)/2.0, 3.0/2.0 + std::sqrt(21.0)/2.0 };
}

template<typename T>
matrix<std::complex<T>> herm_mat_2d_2() {
    auto mat = matrix<std::complex<T>>(2,2);
    mat(0,0) = std::complex<T>(1.0,0.0);
    mat(0,1) = std::complex<T>(1.0, 2.0);
    mat(1,0) = std::complex<T>(1.0, -2.0);
    mat(1,1) = std::complex<T>(0.5, 0.0);
    return mat;
}

template<typename T>
matrix<std::complex<T>> herm_mat_2d_eigenvectors_2() {
    auto mat = matrix<std::complex<T>>(2,2);
    mat(0,0) = std::complex<T>(0.5, -1.0);
    mat(1,0) = std::complex<T>(1.0, 0.0);
    mat(0,1) = std::complex<T>(-0.4, 0.8);
    mat(1,1) = std::complex<T>(1.0, 0.0);
    return mat;
}

template<typename T>
std::vector<T> herm_mat_2d_eigenvalues_2(){
    return { -(3.0/2.0), 3.0 };
}

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

TEST_CASE("Basic Matrix Ops", "[matrix]") {
    auto mat = matrix<std::complex<double>>(2,2);
    mat(0,0) = std::complex<double>(1.0, 0.0);
    mat(0,1) = std::complex<double>(1.0, -2.0);
    mat(1,0) = std::complex<double>(1.0, 2.0);
    mat(1,1) = std::complex<double>(0.5, 0.0);

    SECTION("matrix - col_index") {
        std::vector<std::complex<double>> col0{ {1.0, 0.0}, {1.0, 2.0} };
        std::vector<std::complex<double>> col1{ {1.0, -2.0}, {0.5, 0.0} };

        REQUIRE(AER::Linalg::almost_equal(col0, mat.col_index(0)));
        REQUIRE(AER::Linalg::almost_equal(col1, mat.col_index(1)));
    }

    SECTION("matrix - col_index") {
        std::vector<std::complex<double>> row0{ {1.0, 0.0}, {1.0, -2.0} };
        std::vector<std::complex<double>> row1{ {1.0, 2.0}, {0.5, 0.0} };

        REQUIRE(AER::Linalg::almost_equal(row0, mat.row_index(0)));
        REQUIRE(AER::Linalg::almost_equal(row1, mat.row_index(1)));
    }

    std::cout << "row/col_index working as expected" << std::endl;
}
 
TEST_CASE("Linear Algebra utilities", "[eigen_psd]") {
    SECTION("zheevx - deterministic hermitian matrix - 2") {
        auto herm_mat = herm_mat_2d_2<double>();
        auto expected_eigenvalues = herm_mat_2d_eigenvalues_2<double>();
        auto expected_eigenvectors = herm_mat_2d_eigenvectors_2<double>();

        SECTION("sanity check - eigenvals/vecs should recreate original") {
            // sanity check
            matrix<std::complex<double>> sanity_value(herm_mat.size());
            for (size_t j=0; j < expected_eigenvalues.size(); j++) {
                sanity_value += expected_eigenvalues[j] * AER::Utils::projector(expected_eigenvectors.col_index(j));
            }
            REQUIRE(AER::Linalg::almost_equal(herm_mat, sanity_value));
        }
        SECTION("actual check - heevx returns correctly") {
            std::vector<double> eigenvalues;
            matrix<std::complex<double>> eigenvectors{herm_mat};
            eigensystem_psd_heevx(herm_mat, eigenvalues, eigenvectors);

            // test equality
            REQUIRE(AER::Linalg::almost_equal(expected_eigenvectors, eigenvectors));
            REQUIRE(AER::Linalg::almost_equal(expected_eigenvalues, eigenvalues));

            // test reconstruction
            matrix<std::complex<double>> value(herm_mat.size());
            for (size_t j=0; j < eigenvalues.size(); j++) {
                value += AER::Utils::projector(eigenvectors.col_index(j)) * eigenvalues[j];
            }
            REQUIRE(AER::Linalg::almost_equal(herm_mat, value));
        }
    }

    SECTION("zheevx - random hermitian matrix") {
        auto rand_herm_mat = rand_hermitian_mat<double>(3, 10.0);
        //std::cout << "random hermitian matrix:" << std::endl;
        //std::cout << rand_herm_mat << std::endl;
        std::vector<double> eigenvalues;
        matrix<std::complex<double>> eigenvectors = rand_herm_mat;

        eigensystem_psd_heevx(rand_herm_mat, eigenvalues, eigenvectors);
 
        matrix<std::complex<double>> value(rand_herm_mat.size());

        for (size_t j=0; j < eigenvalues.size(); j++) {
            value += eigenvalues[j] * AER::Utils::projector(eigenvectors.col_index(j));
        }
        REQUIRE(AER::Linalg::almost_equal(rand_herm_mat, value));
    }

/*
    SECTION("zhetrd - random hermitian matrix") {
        auto rand_psd_mat = rand_hermitian_mat<double>(5, 5);
        std::vector<double> eigenvalues;
        matrix<std::complex<double>> eigenvectors = rand_psd_mat;

        eigensystem_psd_hetrd(rand_psd_mat, eigenvalues, eigenvectors);
 
        matrix<std::complex<double>> value(rand_psd_mat.size());

        for (size_t j=0; j < eigenvalues.size(); j++) {
            value += eigenvalues[j] * AER::Utils::projector(eigenvectors.row_index(j));
        }
        REQUIRE(AER::Linalg::almost_equal(rand_psd_mat, value));
    }
*/
/*
    SECTION("the input matrix of complex of doubles, is a PSD"){
        auto psd_matrix_double = create_psd_matrix<double>();

        auto expected_eigenvalues = create_expected_eigenvalues<double>();
        auto expected_eigenvectors = create_expected_eigenvectors<double>();
        std::vector<double> eigenvalues;
        matrix<std::complex<double>> eigenvectors = psd_matrix_double;
        eigensystem_psd(psd_matrix_double, eigenvalues, eigenvectors);

        std::cout << expected_eigenvectors << std::endl;
        std::cout << eigenvectors << std::endl;
        REQUIRE(AER::Linalg::almost_equal(expected_eigenvectors, eigenvectors));
    }
*/
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
    }*/
}


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
            std::numeric_limits<double>::epsilon() + double(1.0),
            std::numeric_limits<double>::epsilon() + double(1.0)
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
