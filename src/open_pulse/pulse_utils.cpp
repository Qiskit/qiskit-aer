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

    int nrows = vec.shape(0);
    complex_t temp, expt = 0;
    int row, jj;

    for (row = 0; row < nrows; row++) {
        temp = 0;
        auto vec_conj = std::conj(vec_raw[row]);
        for (jj = ptr_raw[row]; jj < ptr_raw[row + 1]; jj++) {
            temp += data_raw[jj] * vec_raw[ind_raw[jj]];
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
    for(int i=0; i < meas_size; i++){
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
    unsigned int nrows = mem.shape(0);
    unsigned int nprobs = probs.shape(0);

    unsigned char temp;

    auto mem_raw = mem.mutable_unchecked<2>();
    auto mem_slots_raw = mem_slots.unchecked<1>();
    auto probs_raw = probs.unchecked<1>();
    auto rand_vals_raw = rand_vals.unchecked<1>();

    for(std::size_t ii = 0; ii < nrows; ii++){
        for(std::size_t jj = 0; jj < nprobs; jj++) {
            temp = static_cast<unsigned char>(probs_raw[jj] > rand_vals_raw[nprobs*ii+jj]);
            if(temp) {
                mem_raw(ii, mem_slots_raw[jj]) = temp;
            }
        }
    }
}

void oplist_to_array(py::list A, py::array_t<complex_t> B, int start_idx)
{
    unsigned int lenA = A.size();
    if((start_idx+lenA) > B.shape(0)) {
        throw std::runtime_error(std::string("Input list does not fit into array if start_idx is ") + std::to_string(start_idx) + ".");
    }

    auto B_raw = B.mutable_unchecked<1>();
    for(int kk=0; kk < lenA; kk++){
        auto item = A[kk].cast<py::list>();
        B_raw[start_idx+kk] = complex_t(item[0].cast<double>(), item[1].cast<double>());
    }
}


template <typename T>
T * get_raw_data(py::array_t<T> array)
{
    return static_cast<T *>(array.request().ptr);
}

py::array_t<double> spmv_csr(py::array_t<complex_t> data,
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
    auto out_raw = get_raw_data(vec);
    zspmvpy(data_raw, ind_raw, ptr_raw, vec_raw, 1.0, out_raw, num_rows);

    return out;
}
