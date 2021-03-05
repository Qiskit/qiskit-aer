/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_misc_warnings_hpp_
#define _aer_misc_warnings_hpp_

#if defined(_MSC_VER)
#define DISABLE_WARNING(warningNumber)                                         \
  __pragma(warning(disable : warningNumber))
#define DISABLE_WARNING_PUSH __pragma(warning(push, 0));
#define DISABLE_WARNING_POP __pragma(warning(pop))

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING(warningName)                                           \
  DO_PRAGMA(GCC diagnostic ignored #warningName)
#define DISABLE_WARNING_PUSH                                                   \
  DO_PRAGMA(GCC diagnostic push)                                               \
  DISABLE_WARNING(-Wall)                                                       \
  DISABLE_WARNING(-Wextra)                                                     \
  DISABLE_WARNING(-Wshadow)                                                    \
  DISABLE_WARNING(-Wfloat-equal)                                               \
  DISABLE_WARNING(-Wundef)                                                     \
  DISABLE_WARNING(-Wpedantic)                                                  \
  DISABLE_WARNING(-Wredundant-decls)                                           \
  DISABLE_WARNING(-Wsign-compare)

#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP

#endif

#endif /* Guards */
