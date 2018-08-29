#include <string>

#include "framework/json.hpp"

namespace AER{
namespace Test{
namespace Utilities {
    json_t load_qobj(const std::string& filename){
        return JSON::load(filename);
    }

}