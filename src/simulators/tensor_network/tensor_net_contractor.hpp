/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _tensor_net_contractor_hpp_
#define _tensor_net_contractor_hpp_

namespace AER {
namespace TensorNetwork {

#define AER_TENSOR_NET_MAX_SAMPLING 10000

template <typename data_t = double>
class TensorNetContractor {
protected:
public:
  TensorNetContractor() {}
  virtual ~TensorNetContractor() {}

  virtual void set_device(int idev) = 0;
  virtual void allocate_additional_tensors(uint_t size) = 0;

  virtual void
  set_network(const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
              bool add_sp_tensors = true) = 0;
  virtual void set_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) = 0;
  virtual void set_output(std::vector<int32_t> &modes,
                          std::vector<int64_t> &extents) = 0;

  virtual void update_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) = 0;

  virtual void setup_contraction(bool use_autotune = false) = 0;
  virtual uint_t num_slices(void) = 0;

  virtual void contract(std::vector<std::complex<data_t>> &out) = 0;
  virtual double contract_and_trace(uint_t num_qubits) = 0;

  virtual double contract_and_sample_measure(reg_t &samples,
                                             std::vector<double> &rnds,
                                             uint_t num_qubits) = 0;

  virtual void
  allocate_sampling_buffers(uint_t size = AER_TENSOR_NET_MAX_SAMPLING) = 0;
  virtual void deallocate_sampling_buffers(void) = 0;

  virtual void set_target_gpus(reg_t &t) {}
};

template <typename data_t = double>
class TensorNetContractorDummy : public TensorNetContractor<data_t> {
protected:
public:
  TensorNetContractorDummy() {}
  ~TensorNetContractorDummy() {}

  void set_device(int idev) override {}
  void allocate_additional_tensors(uint_t size) override {}

  void set_network(const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors,
                   bool add_sp_tensors = true) override {}
  void set_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override {}
  void set_output(std::vector<int32_t> &modes,
                  std::vector<int64_t> &extents) override {}

  void update_additional_tensors(
      const std::vector<std::shared_ptr<Tensor<data_t>>> &tensors) override {}

  void setup_contraction(bool use_autotune = false) override {}
  uint_t num_slices(void) override { return 1; }

  void contract(std::vector<std::complex<data_t>> &out) override {}
  double contract_and_trace(uint_t num_qubits) override { return 1.0; }

  double contract_and_sample_measure(reg_t &samples, std::vector<double> &rnds,
                                     uint_t num_qubits) override {
    return 1.0;
  }

  void allocate_sampling_buffers(
      uint_t size = AER_TENSOR_NET_MAX_SAMPLING) override {}
  void deallocate_sampling_buffers(void) override {}
};

//------------------------------------------------------------------------------
} // end namespace TensorNetwork
} // end namespace AER
//------------------------------------------------------------------------------

#endif //_tensor_net_contractor_hpp_
