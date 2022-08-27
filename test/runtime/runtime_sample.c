#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <complex.h>

#include "aer_runtime_api.h"

typedef uint_fast64_t uint_t; 

int main() {
  int num_of_qubits = 8;

  printf("%d-qubits GHZ\n", num_of_qubits);

  void* state = aer_state();

  aer_state_configure(state, "method", "statevector");
  aer_state_configure(state, "device", "CPU");
  aer_state_configure(state, "precision", "double");

  int start_qubit = aer_allocate_qubits(state, num_of_qubits);
  
  aer_state_initialize(state);

  int* qubits = (int*) malloc(sizeof(int) * num_of_qubits);
  for (int i = 0; i < num_of_qubits; ++i)
    qubits[i] = start_qubit + i;

  aer_apply_h(state, qubits[0]);
  for (int i = 0; i < num_of_qubits - 1; ++i)
      aer_apply_cx(state, qubits[i], qubits[i + 1]);

  printf("non-zero probabilities and amplitudes:\n");
  for (uint_t i = 0; i < (1UL << num_of_qubits); ++i) {
    double prob = aer_probability(state, i);
    if (prob > 0.0) {
      double complex amp = aer_amplitude(state, i);
      printf("  %08llx %lf [%lf, %lf]\n", i, prob, creal(amp), cimag(amp));
    }
  }
  
  printf("all the amplitudes:\n");
  double complex* sv = aer_release_statevector(state);
  for (uint_t i = 0; i < (1UL << num_of_qubits); ++i) {
      printf("  %08llx [%lf, %lf]\n", i, creal(sv[i]), cimag(sv[i]));
  }

  aer_state_finalize(state);

  free(qubits);
  free(sv);

  return 1;
}