#define CATCH_CONFIG_MAIN
#include <map>
#include <catch2/catch.hpp>
#include <cmath>
#include <limits>
#include "framework/linalg/almost_equal.hpp"
#include "utils.hpp"

namespace AER{
namespace Test{


TEST_CASE( "Framework Utilities", "[almost_equal]" ) {
    SECTION( "The maximum difference between two numbers over 1.0 is greater than epsilon, so they are amlmost equal" ) {
        double first = 1.0 + std::numeric_limits<double>::epsilon();
        double actual = 1.0;
        // Because the max_diff param is bigger than epsilon, this should be almost equal
        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-15, 1e-15));
        
    }

    SECTION( "The difference between two numbers really close to 0 should say are almost equal" ) {
        double first = 5e-323; // Really close to the min magnitude of double
        double actual = 6e-323;
        REQUIRE(Linalg::almost_equal<decltype(first)>(first, actual, 1e-323, 1e-323));
        
    }
}


//------------------------------------------------------------------------------
} // end namespace Test
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
