////////////////////////////////////////////////////////////////////////////////
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <nlohmann_json.hpp>
#include <iostream>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////

namespace py = pybind11;
namespace nl = nlohmann;

using namespace pybind11::literals;

template <typename T>
json_t numpy_to_json_1d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dims must be 1");
    }
    //std::cout << "1-d conversion: " << std::endl;

    T *ptr = (T *) buf.ptr;
    size_t X = buf.shape[0];

    std::vector<T> tbr;
    for (size_t idx = 0; idx < X; idx++)
        tbr.push_back(ptr[idx]);

    return tbr;
}

template <typename T>
json_t numpy_to_json_2d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dims must be 2");
    }

    T *ptr = (T *) buf.ptr;
    size_t X = buf.shape[0];
    size_t Y = buf.shape[1];

    //std::cout << "2-d conversion: " << X << "x" << Y << std::endl;

    std::vector<std::vector<T > > tbr;
    for (size_t idx = 0; idx < X; idx++) {
        std::vector<T> tbr1;
        for (size_t jdx = 0; jdx < Y; jdx++) {
            tbr1.push_back(ptr[idx + X*jdx]);
        }
        tbr.push_back(tbr1);
    }

    return tbr;

}

template <typename T>
json_t numpy_to_json_3d(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Number of dims must be 3");
    }
    T *ptr = (T *) buf.ptr;
    size_t X = buf.shape[0];
    size_t Y = buf.shape[1];
    size_t Z = buf.shape[2];

    //std::cout << "3-d conversion: " << X << "x" << Y << "x" << Z << std::endl;

    std::vector<std::vector<std::vector<T > > > tbr;
    for (size_t idx = 0; idx < X; idx++) {
        std::vector<std::vector<T> > tbr1;
        for (size_t jdx = 0; jdx < Y; jdx++) {
            std::vector<T> tbr2;
            for (size_t kdx = 0; kdx < Z; kdx++) {
                tbr2.push_back(ptr[kdx + Z*(jdx + Y*idx)]);
            }
            tbr1.push_back(tbr2);
        }
        tbr.push_back(tbr1);
    }

    return tbr;

}

template <typename T>
json_t numpy_to_json(py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    //std::cout << "buff dim: " << buf.ndim << std::endl;

    if (buf.ndim == 1) {
        return numpy_to_json_1d(arr);
    } else if (buf.ndim == 2) {
        return numpy_to_json_2d(arr);
    } else if (buf.ndim == 3) {
        return numpy_to_json_3d(arr);
    } else {
        throw std::runtime_error("Invalid number of dimensions!");
    }
    json_t tbr;
    return tbr;
}

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
                    obj.append(from_json_impl(el));
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
            //std::cout << "Casting..." << obj.cast<py::object>() << std::endl;
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
                for (auto item : py::cast<py::dict>(obj))
                {
                    out[item.first.cast<nl::json::string_t>()] = to_json_impl(item.second);
                }
                return out;
            }
            if (py::isinstance<py::array_t<double> >(obj))
            {
                return numpy_to_json(obj.cast<py::array_t<double, py::array::c_style> >());
            }
            if (py::isinstance<py::array_t<std::complex<double> > >(obj))
            {
                return numpy_to_json(obj.cast<py::array_t<std::complex<double>, py::array::c_style> >());
            } 
            if (std::string(py::str(obj.get_type())) == "<class \'complex\'>")
            {
                auto tmp = obj.cast<std::complex<double>>();
                json out;
                out.push_back(tmp.real());
                out.push_back(tmp.imag());
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
