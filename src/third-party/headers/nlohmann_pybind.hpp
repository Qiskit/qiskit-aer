////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>

#include <nlohmann_json.hpp>

////////////////////////////////////////////////////////////////////////////////

namespace py = pybind11;
namespace nl = nlohmann;

using namespace pybind11::literals;

namespace nlohmann
{

    namespace detail
    {
        py::object from_json_impl(const json& j)
        {
            if (j.is_null())
            {
                return py::none();
            }
            if (j.is_boolean())
            {
                return py::bool_(j.get<nl::json::boolean_t>());
            }
            if (j.is_number())
            {
                if (j.is_number_float()) {
                    return py::float_(j.get<nl::json::number_float_t>());
                } else if (j.is_number_unsigned()) {
                    return py::int_(j.get<nl::json::number_unsigned_t>());
                } else {
                    return py::int_(j.get<nl::json::number_integer_t>());
                }
            }
            if (j.is_string())
            {
                return py::str(j.get<nl::json::string_t>());
            }
            if (j.is_array())
            {
                py::list obj;
                for (const auto& el: j)
                {
                    obj.attr("append")(from_json_impl(el));
                }
                return obj;
            }
            if (j.is_object())
            {
                py::dict obj;
                for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it)
                {
                    obj[py::str(it.key())] = from_json_impl(it.value());
                }
                return obj;
            }
        }

        json to_json_impl(py::handle obj)
        {
            if (obj.is_none())
            {
                return nullptr;
            }
            if (py::isinstance<py::bool_>(obj))
            {
                return obj.cast<nl::json::boolean_t>();
            }
            if (py::isinstance<py::int_>(obj))
            {
                return obj.cast<nl::json::number_integer_t>();
            }
            if (py::isinstance<py::float_>(obj))
            {
                return obj.cast<nl::json::number_float_t>();
            }
            if (py::isinstance<py::str>(obj))
            {
                return obj.cast<nl::json::string_t>();
            }
            if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj))
            {
                json out;
                for (py::handle value: obj)
                {
                    out.push_back(to_json_impl(value));
                }
                return out;
            }
            if (py::isinstance<py::dict>(obj))
            {
                json out;
                for (py::handle key: obj)
                {
                    out[key.cast<nl::json::string_t>()] = to_json_impl(obj[key]);
                }
                return out;
            }
            throw std::runtime_error("to_json not implemented for this type of object: " + obj.cast<std::string>());
        }
    }

    template <>
    struct adl_serializer<py::object>
    {
        static py::object from_json(const json& j)
        {
            return detail::from_json_impl(j);
        }

        static void to_json(json& j, const py::object& obj)
        {
            j = detail::to_json_impl(obj);
        }
    };
}
