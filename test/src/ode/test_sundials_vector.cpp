#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <valarray>

#include <ode/sundials_wrapper/sundials_complex_vector.hpp>

namespace AER {
  namespace Test {
    // Use valarray to handle math with vectors
    std::valarray<complex_t> v{{2., 4.2},
                               {1., 4.3},
                               {2.4, 5.1}};

    std::valarray<complex_t> w{{1.3, 3.4},
                               {2.1, 5.5},
                               {3.3, 6.6}};

    std::valarray<complex_t> weights{{2.,0.}, {1.,0.}, {3.,0.}};

    double a = 3;
    double b = 5;

    using SunComplexContent_t = SundialsComplexContent<std::vector<complex_t>>;
    namespace{
      std::vector<complex_t> to_vector(const std::valarray<complex_t>& va){
        std::vector<complex_t> ret;
        ret.assign(std::begin(va), std::end(va));
        return ret;
      }

      std::valarray<double> val_norm(const std::valarray<complex_t>& input){
        std::valarray<double> ret(input.size());
        for(size_t i=0; i<input.size(); ++i){
          ret[i] = std::norm(input[i]);
        }
        return ret;
      }
    }

    TEST_CASE( "Sundials Vector creation with length", "[sundials vector]" ) {
      auto sun_vector_v = SunComplexContent_t::new_vector(4);
      REQUIRE(SunComplexContent_t::get_data(sun_vector_v).size() == 4);
      REQUIRE(SundialsOps<SunComplexContent_t>::SundialsComplexContent_GetVectorID(sun_vector_v) == N_Vector_ID::SUNDIALS_NVEC_CUSTOM);
    }

    TEST_CASE( "Sundials Vector creation from vector", "[sundials vector]" ) {
      auto vec_v = to_vector(v);
      auto sun_vector_v = SunComplexContent_t::new_vector(vec_v);
      REQUIRE(SunComplexContent_t::get_data(sun_vector_v) == vec_v);
    }

    TEST_CASE( "Sundials Vector clone empty", "[sundials vector]" ) {
      auto vec_v = to_vector(v);
      auto sun_vector_v = SunComplexContent_t::new_vector(vec_v);
      auto sun_vector_cloned = SundialsOps<SunComplexContent_t>::SundialsComplexContent_CloneEmpty(sun_vector_v);

      REQUIRE(SunComplexContent_t::get_data(sun_vector_v) == vec_v);
    }

    TEST_CASE( "Sundials Vector destruction", "[sundials vector]" ) {
      auto sun_vector_v = SunComplexContent_t::new_vector(to_vector(v));
      SundialsOps<SunComplexContent_t>::SundialsComplexContent_Destroy(sun_vector_v);
      REQUIRE(SunComplexContent_t::get_content(sun_vector_v) == nullptr);
    }

    TEST_CASE( "Get size", "[sundials vector]" ) {
      auto vec_v = to_vector(v);
      auto sun_vector_v = SunComplexContent_t::new_vector(vec_v);
      REQUIRE(SunComplexContent_t::get_size(sun_vector_v) == vec_v.size());
    }

    TEST_CASE("Sundials unary operations", "[unitary operations]"){
      auto sun_vector_v = SunComplexContent_t::new_vector(v);
      auto vec_v = to_vector(v);

      SECTION("Sundials set constant"){
        SundialsOps<SunComplexContent_t>::SundialsComplexContent_Const(a, sun_vector_v);
        REQUIRE(SunComplexContent_t::get_data(sun_vector_v) == to_vector(0. * v + a));
      }

      SECTION("Sundials max norm"){
        auto norm = SundialsOps<SunComplexContent_t>::SundialsComplexContent_MaxNorm(sun_vector_v);
        auto result = val_norm(v);
        REQUIRE(norm == std::sqrt(result.max()));
      }

      SECTION("Sundials min real part"){
        auto min_real = SundialsOps<SunComplexContent_t>::SundialsComplexContent_Min(sun_vector_v);
        REQUIRE(min_real == 1);
      }
    }

    TEST_CASE("Sundials unary operations with vector output", "[binary operations]"){
      auto sun_vector_v = SunComplexContent_t::new_vector(to_vector(v));
      auto sun_vector_res = SunComplexContent_t::new_vector(static_cast<int>(w.size()));

      SECTION("Sundials scale"){
        SundialsOps<SunComplexContent_t>::SundialsComplexContent_Scale(a, sun_vector_v, sun_vector_res);
        REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(a * v));
      }

      SECTION("Sundials absolute value"){
        SundialsOps<SunComplexContent_t>::SundialsComplexContent_Abs(sun_vector_v, sun_vector_res);
        REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(std::abs(v)));
      }

      SECTION("Sundials inverse"){
        SundialsOps<SunComplexContent_t>::SundialsComplexContent_Inv(sun_vector_v, sun_vector_res);
        REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(1./v));
      }

      SECTION("Sundials add const"){
        SundialsOps<SunComplexContent_t>::SundialsComplexContent_AddConst(sun_vector_v, a, sun_vector_res);
        REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(a + v));
      }
    }

    TEST_CASE("Sundials binary operations", "[binary operations]"){
      auto sun_vector_v = SunComplexContent_t::new_vector(to_vector(v));
      //auto sun_vector_w = SunComplexContent_t::new_vector(to_vector(w));

      SECTION("Sundials weighted norm"){
        auto sun_vector_weights = SunComplexContent_t::new_vector(w);
        auto wrmsnorm = SundialsOps<SunComplexContent_t>::SundialsComplexContent_WrmsNorm(sun_vector_v, sun_vector_weights);
        auto result = val_norm(v*w);
        REQUIRE(wrmsnorm == std::sqrt(result.sum()/result.size()));
      }
    }

    TEST_CASE("Sundials binary operations with vector output", "[binary operations]"){
      auto sun_vector_v = SunComplexContent_t::new_vector(to_vector(v));
      auto sun_vector_w = SunComplexContent_t::new_vector(to_vector(w));
      auto sun_vector_res = SunComplexContent_t::new_vector(static_cast<int>(w.size()));

      SECTION("Sundials Vector linear add"){
        SECTION("Linear add (v+w)"){
          SundialsOps<SunComplexContent_t>::SundialsComplexContent_LinearSum(1.0, sun_vector_v, 1.0, sun_vector_w, sun_vector_res);
          REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(v + w));
        }
        SECTION("Scaled Linear add (a*v + b*w)"){
          SundialsOps<SunComplexContent_t>::SundialsComplexContent_LinearSum(a, sun_vector_v, b, sun_vector_w, sun_vector_res);
          REQUIRE(SunComplexContent_t::get_data(sun_vector_res) == to_vector(a*v + b*w));
        }
      }
    }
  }
}

