#include <string>
#include "framework/json.hpp"

namespace AER{
namespace Test{
namespace Utilities {
    inline json_t load_qobj(const std::string& filename){
        return JSON::load(filename);
    }

    template<typename T>
    T calculate_floats(T start, T decrement, int count){
        for (int i = 0; i < count; ++i)
            start -= decrement;
        return start;
    }

} // End of Utilities namespace
} // End of Test namespace
} // End of AER namspace