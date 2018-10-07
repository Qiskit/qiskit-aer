#include <string>

#include "framework/json.hpp"

namespace AER{
namespace Test{
namespace Utilities {
    inline json_t load_qobj(const std::string& filename){
        return JSON::load(filename);
    }

} // End of Utilities namespace
} // End of Test namespace
} // End of AER namspace