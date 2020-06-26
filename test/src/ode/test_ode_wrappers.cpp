#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <vector>
#include <framework/types.hpp>
#include <framework/linalg/almost_equal.hpp>
#include <ode/sundials_wrappers/sundials_cvode_wrapper.hpp>
#include <ode/odeint_wrappers/abm_wrapper.hpp>

namespace Catch {
  std::string convertMyTypeToString(const std::vector<AER::complex_t> &value) {
    std::ostringstream oss;
    oss << "{ " << std::setprecision(15);
    for (const auto &elem : value) {
      oss << elem;
    }
    oss << " }\n";
    return oss.str();
  }

  template <> struct StringMaker<std::vector<AER::complex_t>> {
    static std::string convert(const std::vector<AER::complex_t> &value) {
      return convertMyTypeToString(value);
    }
  };
} // namespace Catch

namespace AER {
  namespace ODE {
    namespace Test {
      namespace {

        template <typename T>
        void rhs_in_to_bind(double t, const T &y_raw, T &ydot_raw, double par) {
          ydot_raw[0] = par * y_raw[1];
          ydot_raw[1] = -y_raw[0] + y_raw[1] + y_raw[4];
          ydot_raw[2] = y_raw[1] + y_raw[2];
          ydot_raw[3] = y_raw[3] - y_raw[4];
          ydot_raw[4] = y_raw[0] + y_raw[4];
        }

        template <typename T>
        void rhs_in_to_bind_sens(double t, const T &y_raw, T &ydot_raw, double par1, double par2) {
          ydot_raw[0] = par1 * y_raw[1];
          ydot_raw[1] = -y_raw[0] + y_raw[1] + y_raw[4];
          ydot_raw[2] = y_raw[1] + par2 * y_raw[2];
          ydot_raw[3] = y_raw[3] - y_raw[4];
          ydot_raw[4] = y_raw[3] + y_raw[4];
        }
        std::vector<complex_t> y{{1, 1}, {3, 4}, {1, -1}, {0, 5}, {3, -2.6}};
        std::vector<complex_t> y_sens{{1, 0}, {3, 0}, {1, 0}, {0, 0}, {3, -0}};

        inline bool compare(const std::vector<complex_t> &lhs, const std::vector<complex_t> &rhs,
                            double prec = 1e-5) {
          if (lhs.size() != rhs.size())
            return false;
          for (size_t i = 0; i < lhs.size(); ++i) {
            if (!(Linalg::almost_equal(lhs[i], rhs[i], prec, prec))) {
              std::cout << "Vectors differ at element: " << i << std::setprecision(22) << ". [" << lhs[i]
                        << "] != [" << rhs[i] << "]\n";
              std::cout << "Vectors differ: " << Catch::convertMyTypeToString(lhs)
                        << " != " << Catch::convertMyTypeToString(rhs) << std::endl;
              return false;
            }
          }
          return true;
        }

        template <typename T>
        T getODE() {
          auto f = [](double t, const std::vector<complex_t> &y_raw,
                      std::vector<complex_t> &ydot_raw) {
            rhs_in_to_bind<std::vector<complex_t>>(t, y_raw, ydot_raw, -5.);
          };
          auto ret = T(f, y, 0.0);
          ret.set_tolerances(1e-12, 1e-6);
          ret.set_step_limits(10, 0, 0.01);
          return ret;
        }
      } // namespace

      TEMPLATE_TEST_CASE("C++ ODE wrappers integration", "[c++ wrappers]",
                         CvodeWrapper<std::vector<complex_t>>, ABMWrapper<std::vector<complex_t>>) {
        auto ode = getODE<TestType>();
        constexpr double finalTime1 = .12;
        constexpr double finalTime2 = .24;
        Catch::StringMaker<double>::precision = 15;
        std::vector<complex_t> result_t1{{-1.01633508699953, -1.44150979311768},
                                         {3.78483441012611, 4.18372211652272},
                                         {1.55496015991325, -0.608921922058043},
                                         {-0.408859454752513, 5.98761842790629},
                                         {3.38900655901267, -2.95536026241699}};

        std::vector<complex_t> result_t2{{-3.6270250806984,-4.07587275367097},
                                         {4.99676915888355,4.64491814764456},
                                         {2.30642935895291,-0.127355964210215},
                                         {-0.905395125199613,7.16866922595626},
                                         {3.53612132708047,-3.67759232211879}};

        SECTION("C++ ODE wrapper initialization") {
          REQUIRE(ode.get_solution() == y);
        }

        SECTION("integrate till finalTime=" + std::to_string(finalTime1)) {
          ode.integrate(finalTime1);
          REQUIRE(
              compare(ode.get_solution(), result_t1));
        }

        SECTION("integrate till finalTime=" + std::to_string(finalTime2)) {
          ode.integrate(finalTime2);
          REQUIRE(compare(ode.get_solution(), result_t2));
        }

        SECTION("2 Integrators till finalTime=" + std::to_string(finalTime1)) {
          auto ode2 = getODE<TestType>();
          ode2.set_tolerances(1e-12, 1e-6);
          ode2.set_step_limits(10, 0, 0.01);
          ode.integrate(finalTime1);
          ode2.integrate(finalTime1);
          REQUIRE(compare(ode.get_solution(), result_t1));
          REQUIRE(compare(ode2.get_solution(), result_t1));
        }

        SECTION("Move integrator and integrate to time " + std::to_string(finalTime1)) {
          auto ode2 = std::move(ode);
          ode2.integrate(finalTime1);
          REQUIRE(compare(ode2.get_solution(), result_t1));
        }

        SECTION("RHS bind and integrate to time" + std::to_string(finalTime1)) {
          auto ode2 = getODE<TestType>();
          ode2.integrate(finalTime1);
          REQUIRE(compare(ode2.get_solution(), result_t1));
          auto ode3 = std::move(ode2);
          ode3.integrate(finalTime1);
          REQUIRE(compare(ode3.get_solution(), result_t1));
        }

        SECTION("Change settings after integrate") {
          ode.integrate(finalTime1);
          ode.set_tolerances(1e-6, 1e-6);
          ode.set_max_nsteps(15000);
          ode.set_maximum_order(5);
          ode.set_step_limits(10, 0, 0.005);
          ode.integrate(finalTime2);
          REQUIRE(compare(ode.get_solution(), result_t2, 1e-2));
        }
      }

      TEST_CASE("Sensitivities computation", "[sundials cvode]") {
        SECTION("RHS with 2 params"){
          double par1 = -5;
          double par2 = -2;
          rhsFuncType<std::vector<complex_t>> f = [&par1, &par2](double t,
                                                                 const std::vector<complex_t> &y_raw,
                                                                 std::vector<complex_t> &ydot_raw) {
            return rhs_in_to_bind_sens<std::vector<complex_t>>(t, y_raw, ydot_raw, par1, par2);
          };
          std::vector<double> p{par1, par2};
          auto pf = [&par1, &par2](const std::vector<double> &p) {
            par1 = p[0];
            par2 = p[1];
          };
          auto ode_sens = CvodeWrapper<std::vector<complex_t>>(f, y_sens, 0.0);
          ode_sens.setup_sens(pf, p);
          ode_sens.set_tolerances(1e-12, 1e-6);
          constexpr double finalTime1 = .12;
          Catch::StringMaker<double>::precision = 15;

          SECTION("Intialize for sensitivities computation") {
            REQUIRE(ode_sens.get_solution() == y_sens);
          }

          SECTION("Sensitivity ODE integrate till finalTime=" + std::to_string(finalTime1)) {
            ode_sens.integrate(finalTime1);
            REQUIRE(
                compare(ode_sens.get_sens_solution(0), std::vector<complex_t>{{0.40778922348274, 0},
                                                                              {-0.0243315050031365, 0},
                                                                              {-0.000890227247355051, 0},
                                                                              {0, 0},
                                                                              {0, 0}}));
            REQUIRE(compare(ode_sens.get_sens_solution(1),
                            std::vector<complex_t>{{0, 0}, {0, 0}, {0.1143010069399, 0}, {0, 0}, {0, 0}}));
            CHECK_THROWS(ode_sens.get_sens_solution(2));
          }
        }

        SECTION("Sensitivities computation. Movement.") {
          double par = 5;
          rhsFuncType<std::vector<complex_t>> f =
              [&par](double t, const std::vector<complex_t> &y_raw,
                     std::vector<complex_t> &ydot_raw) { ydot_raw[0] = par * y_raw[0]; };
          std::vector<double> p{par};
          auto pf = [&par](const std::vector<double> &p) { par = p[0]; };
          std::vector<complex_t> y_sens_m{{3.5, 0}};
          auto ode_sens = CvodeWrapper<std::vector<complex_t>>(f, y_sens_m, 0.0);
          ode_sens.setup_sens(pf, p);
          ode_sens.set_tolerances(1e-12, 1e-6);
          auto ode_sens_2 = std::move(ode_sens);
          ode_sens_2.integrate(0.1);
          REQUIRE(compare(ode_sens_2.get_sens_solution(0),
                          std::vector<complex_t>{{0.5770524532543648277283, 0}}));
        }
      }
    } // namespace Test
  } // namespace ODE
} // namespace AER
