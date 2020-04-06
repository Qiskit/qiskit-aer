#include "pulse_utils.hpp"
#include "zspmv.hpp"

complex_t internal_expect_psi_csr(const py::array_t<complex_t>& data,
                                  const py::array_t<int>& ind,
                                  const py::array_t<int>& ptr,
                                  const py::array_t<complex_t>& vec) {
    auto data_raw = data.unchecked<1>();
    auto vec_raw = vec.unchecked<1>();
    auto ind_raw = ind.unchecked<1>();
    auto ptr_raw = ptr.unchecked<1>();

    auto nrows = vec.shape(0);
    complex_t temp, expt = 0;

    for (decltype(nrows) row = 0; row < nrows; row++) {
        temp = 0;
        auto vec_conj = std::conj(vec_raw[row]);
        for (auto j = ptr_raw[row]; j < ptr_raw[row + 1]; j++) {
            temp += data_raw[j] * vec_raw[ind_raw[j]];
        }
        expt += vec_conj * temp;
    }
    return expt;
}


py::object expect_psi_csr(py::array_t<complex_t> data,
                          py::array_t<int> ind,
                          py::array_t<int> ptr,
                          py::array_t<complex_t> vec,
                          bool isherm){
    complex_t expt = internal_expect_psi_csr(data, ind, ptr, vec);
    if(isherm){
        return py::cast(std::real(expt));
    }
    return py::cast(expt);
}


py::array_t<double> occ_probabilities(py::array_t<int> qubits,
                                      py::array_t<complex_t> state,
                                      py::list meas_ops){
    auto meas_size = meas_ops.size();
    py::array_t<double> probs(meas_size);
    auto probs_raw = probs.mutable_unchecked<1>();
    for(decltype(meas_size) i=0; i < meas_size; i++){
        auto data = meas_ops[i].attr("data").attr("data").cast<py::array_t<complex_t>>();
        auto ind = meas_ops[i].attr("data").attr("indices").cast<py::array_t<int>>();
        auto ptr = meas_ops[i].attr("data").attr("indptr").cast<py::array_t<int>>();

        probs_raw[i] = std::real(internal_expect_psi_csr(data, ind, ptr, state));
    }

    return probs;
}

void write_shots_memory(py::array_t<unsigned char> mem,
                        py::array_t<unsigned int> mem_slots,
                        py::array_t<double> probs,
                        py::array_t<double> rand_vals)
{
    auto nrows = mem.shape(0);
    auto nprobs = probs.shape(0);

    unsigned char temp;

    auto mem_raw = mem.mutable_unchecked<2>();
    auto mem_slots_raw = mem_slots.unchecked<1>();
    auto probs_raw = probs.unchecked<1>();
    auto rand_vals_raw = rand_vals.unchecked<1>();

    for(decltype(nrows) i = 0; i < nrows; i++){
        for(decltype(nprobs) j = 0; j < nprobs; j++) {
            temp = static_cast<unsigned char>(probs_raw[j] > rand_vals_raw[nprobs*i+j]);
            if(temp) {
                mem_raw(i, mem_slots_raw[j]) = temp;
            }
        }
    }
}

void oplist_to_array(py::list A, py::array_t<complex_t> B, int start_idx)
{
    auto lenA = A.size();
    if((start_idx+lenA) > B.shape(0)) {
        throw std::runtime_error(std::string("Input list does not fit into array if start_idx is ") + std::to_string(start_idx) + ".");
    }

    auto B_raw = B.mutable_unchecked<1>();
    for(decltype(lenA) kk=0; kk < lenA; kk++){
        auto item = A[kk].cast<py::list>();
        B_raw[start_idx+kk] = complex_t(item[0].cast<double>(), item[1].cast<double>());
    }
}


template <typename T>
T * get_raw_data(py::array_t<T> array)
{
    return static_cast<T *>(array.request().ptr);
}

py::array_t<complex_t> spmv_csr(py::array_t<complex_t> data,
                                py::array_t<int> ind,
                                py::array_t<int> ptr,
                                py::array_t<complex_t> vec)
{
    auto data_raw = get_raw_data(data);
    auto ind_raw = get_raw_data(ind);
    auto ptr_raw = get_raw_data(ptr);
    auto vec_raw = get_raw_data(vec);

    auto num_rows = vec.shape(0);

    py::array_t<complex_t> out(num_rows);
    auto out_raw = get_raw_data(out);
    memset(&out_raw[0], 0, num_rows * sizeof(complex_t));
    zspmvpy(data_raw, ind_raw, ptr_raw, vec_raw, 1.0, out_raw, num_rows);

    return out;
}
