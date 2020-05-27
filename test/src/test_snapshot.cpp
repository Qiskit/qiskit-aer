#define CATCH_CONFIG_MAIN
#include <map>
#include <catch2/catch.hpp>

#include <controllers/qasm_controller.hpp>

#include "utils.hpp"

namespace AER{
namespace Test{

TEST_CASE( "Simulators Snapshot", "[snaphot]" ) {
    std::map<std::string, json_t> qobj_snapshots;
    qobj_snapshots["state"] =
        AER::Test::Utilities::load_qobj("../../test/data/qobj_snapshot_statevector.json");
    qobj_snapshots["probs"] =
        AER::Test::Utilities::load_qobj("../../test/data/qobj_snapshot_probs.json");
    qobj_snapshots["pauli"] =
        AER::Test::Utilities::load_qobj("../../test/data/qobj_snapshot_expval_pauli.json");
    qobj_snapshots["matrix"] =
        AER::Test::Utilities::load_qobj("../../test/data/qobj_snapshot_expval_matrix.json");

    AER::Simulator::QasmController sim{};

    SECTION( "State simulator snapshot" ) {
        auto expected_result = R"({
                "final":[[[0.7071067811865476,0.0],[0.0,0.0],[0.0,0.0],[0.7071067811865475,0.0]]],
                "initial":[[[1.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]],
                "middle":[[[0.7071067811865476,0.0],[0.7071067811865475,0.0],[0.0,0.0],[0.0,0.0]]]
            })"_json;
        auto result = sim.execute(qobj_snapshots["state"]);
        result = result["results"][0]["data"]["snapshots"]["state"];
        REQUIRE(result == expected_result);
    }
    SECTION( "Probs simulator snapshot" ) {
        REQUIRE(false);
    }
    SECTION( "Pauli simulator snaphsot" ) {
        REQUIRE(false);
    }
    SECTION( "Unitary simulator snapshot" ) {
        REQUIRE(false);
    }
}

//------------------------------------------------------------------------------
} // end namespace Test
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
