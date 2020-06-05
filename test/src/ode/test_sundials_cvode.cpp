#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <ode/sundials_wrapper/sundials_cvode_wrapper.hpp>
#include <vector>
#include <framework/types.hpp>
#include <framework/linalg/almost_equal.hpp>

namespace Catch {
  std::string convertMyTypeToString(const std::vector<AER::complex_t>& value){
    std::ostringstream oss;
    oss << "{ " << std::setprecision(15);
    for(const auto& elem: value){
      oss << elem;
    }
    oss << " }\n";
    return oss.str();
  }

  template<>
  struct StringMaker<std::vector<AER::complex_t>> {
    static std::string convert(const std::vector<AER::complex_t>& value) {
      return convertMyTypeToString(value);
    }
  };
}

namespace AER{
  namespace Test {
    namespace {
      template <typename T>
      void rhs_in(double t, const T& y_raw, T& ydot_raw){
        ydot_raw[0] = -5. * y_raw[1];
        ydot_raw[1] = -y_raw[0] + y_raw[1] + y_raw[4];
        ydot_raw[2] = y_raw[1] + y_raw[2];
        ydot_raw[3] = y_raw[3] - y_raw[4];
        ydot_raw[4] = y_raw[0] + y_raw[4];
      }

      template <typename T>
      void rhs_in_to_bind(double t, const T& y_raw, T& ydot_raw, double par){
        ydot_raw[0] = par * y_raw[1];
        ydot_raw[1] = -y_raw[0] + y_raw[1] + y_raw[4];
        ydot_raw[2] = y_raw[1] + y_raw[2];
        ydot_raw[3] = y_raw[3] - y_raw[4];
        ydot_raw[4] = y_raw[0] + y_raw[4];
      }

      std::vector<complex_t> y{{1,1}, {3,4}, {1,-1}, {0,5}, {3,-2.6}};

      inline bool compare(const std::vector<complex_t>& lhs, const std::vector<complex_t>& rhs, double prec=1e-10){
        if(lhs.size() != rhs.size()) return false;
        for(size_t i = 0; i < lhs.size(); ++i) {
          if(!(Linalg::almost_equal(lhs[i], rhs[i], prec, prec))){
            std::cout << "Vectors differ at element: " << i << std::setprecision(22) << ". [" << lhs[i] << "] != [" << rhs[i] << "]\n";
            std::cout << "Vectors differ: " << Catch::convertMyTypeToString(lhs) << " != " << Catch::convertMyTypeToString(rhs) << std::endl;
            return false;
          }
        }
        return true;
      }

      CvodeWrapper<std::vector<complex_t>> getODE(){
        auto f = std::bind(rhs_in_to_bind<std::vector<complex_t>>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, -5.);
        return CvodeWrapper<std::vector<complex_t>>(OdeMethod::ADAMS, f, y, 0.0, 1e-6, 1e-12);
      }
    }


    TEST_CASE( "Sundials Cvode wrapper initialization", "[sundials cvode]" ) {
      auto ode = CvodeWrapper<std::vector<complex_t>>(OdeMethod::ADAMS, rhs_in<std::vector<complex_t>>, y, 0.0, 1e-6, 1e-12);
      REQUIRE(ode.get_solution() == y);
    }

    TEST_CASE( "Sundials Cvode wrapper ADAMS integration", "[sundials cvode]"){
      auto ode = CvodeWrapper<std::vector<complex_t>>(OdeMethod::ADAMS, rhs_in<std::vector<complex_t>>, y, 0.0, 1e-6, 1e-12);
      constexpr double finalTime1 = .12;
      constexpr double finalTime2 = 12;
      Catch::StringMaker<double>::precision = 15;
      SECTION( "integrate till finalTime=" + std::to_string(finalTime1)){
        ode.integrate(finalTime1);
        REQUIRE( compare(ode.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                              {3.78483441012611,4.18372211652272},
                                                              {1.55496015991325,-0.608921922058043},
                                                              {-0.408859454752513,5.98761842790629},
                                                              {3.38900655901267,-2.95536026241699}}));
      }

      SECTION( "integrate till finalTime=" + std::to_string(finalTime2)){
        ode.integrate(finalTime2);
        REQUIRE( compare(ode.get_solution(), std::vector<complex_t>{{-475328953679.512,-60947969116.8455},
                                                                     {150407152456.807,-10802063608.4835},
                                                                     {186111585748.492,42625869754.6063},
                                                                     {304821334341.734,162985559077.097},
                                                                     {-455227510269.729,-152182453837.88}}));
      }

      SECTION( "2 Integrators" + std::to_string(finalTime1)){
        auto ode2 = CvodeWrapper<std::vector<complex_t>>(OdeMethod::ADAMS, rhs_in<std::vector<complex_t>>, y, 0.0, 1e-6, 1e-12);
        ode.integrate(finalTime1);
        ode2.integrate(finalTime1);

        REQUIRE( compare(ode.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                             {3.78483441012611,4.18372211652272},
                                                             {1.55496015991325,-0.608921922058043},
                                                             {-0.408859454752513,5.98761842790629},
                                                             {3.38900655901267,-2.95536026241699}}));
        REQUIRE( compare(ode2.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                             {3.78483441012611,4.18372211652272},
                                                             {1.55496015991325,-0.608921922058043},
                                                             {-0.408859454752513,5.98761842790629},
                                                             {3.38900655901267,-2.95536026241699}}));
      }

      SECTION( "Move integrator and integrate to time " + std::to_string(finalTime1)){
        auto ode2 = std::move(ode);
        ode2.integrate(finalTime1);
        REQUIRE( compare(ode2.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                                    {3.78483441012611,4.18372211652272},
                                                                    {1.55496015991325,-0.608921922058043},
                                                                    {-0.408859454752513,5.98761842790629},
                                                                    {3.38900655901267,-2.95536026241699}}));
      }

      SECTION("RHS bind and integrate to time" + std::to_string(finalTime1)){
        auto ode2 = getODE();
        ode2.integrate(finalTime1);
        REQUIRE( compare(ode2.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                             {3.78483441012611,4.18372211652272},
                                                             {1.55496015991325,-0.608921922058043},
                                                             {-0.408859454752513,5.98761842790629},
                                                             {3.38900655901267,-2.95536026241699}}));
        auto ode3 = std::move(ode2);
        ode3.integrate(finalTime1);
        REQUIRE( compare(ode3.get_solution(), std::vector<complex_t>{{-1.01633508699953,-1.44150979311768},
                                                             {3.78483441012611,4.18372211652272},
                                                             {1.55496015991325,-0.608921922058043},
                                                             {-0.408859454752513,5.98761842790629},
                                                             {3.38900655901267,-2.95536026241699}}));
      }
    }
  }
}
