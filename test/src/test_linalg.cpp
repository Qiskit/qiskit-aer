#define CATCH_CONFIG_MAIN
#include <map>
#include <catch.hpp>

#include <controllers/qasm_controller.hpp>
#include <framework/linalg/linalg.hpp>

#include "utils.hpp"

namespace AER{
namespace Test{

SCENARIO("Testing linear algebra utilities") {
    GIVEN("The egienvalues and eigenvectors from a positive semi-definitive matrix"){
        matrix<std::complex<float_t>> psd_matrix = {


        };

        WHEN("the input matrix is PSD"){
            auto expected_result = true;

            THEN("eigen_psd() function should return the correct eigenvalues/eigenvectors"){
                auto result = false;
                REQUIRE(result == expected_result);
            }
        }
        WHEN("the input matrix is NOT a PSD"){
            THEN("eigen_psd() function should return incorrect eignevalues/eigenvectors"){
                auto result = false;
                REQUIRE(result == expected_result);
            }
        }

    }
}

//------------------------------------------------------------------------------
} // end namespace Test
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
