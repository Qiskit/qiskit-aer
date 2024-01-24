# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for utility functions to create device noise model.
"""

from test.terra.common import QiskitAerTestCase

from qiskit.providers.aer.backends.controller_wrappers import *


class TestClassicalExpressions(QiskitAerTestCase):
    """Testing classical expressions"""

    def test_eval_uint(self):
        """test eval_uint of uint and bool"""
        self.assertEqual(AerUintValue(32, 0).eval_uint(""), 0)
        self.assertEqual(AerUintValue(32, 1).eval_uint(""), 1)
        self.assertEqual(AerUintValue(32, 10).eval_uint(""), 10)

        try:
            AerBoolValue(False).eval_uint("")
            self.fail("do not reach here")
        except Exception:
            pass

        try:
            AerBoolValue(True).eval_uint("")
            self.fail("do not reach here")
        except Exception:
            pass

    def test_eval_bool(self):
        """test eval_bool of uint and bool"""
        try:
            AerUintValue(32, 0).eval_bool("")
            self.fail("do not reach here")
        except Exception:
            pass

        try:
            AerUintValue(32, 1).eval_bool("")
            self.fail("do not reach here")
        except Exception:
            pass

        self.assertEqual(AerBoolValue(False).eval_bool(""), False)
        self.assertEqual(AerBoolValue(True).eval_bool(""), True)

    def test_var(self):
        """test AerVar"""

        # normal ordering
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("000"), 0)
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("001"), 1)
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("010"), 2)
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("011"), 3)
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("111"), 7)

        # custom ordering
        self.assertEqual(AerVar(AerUint(3), [1, 0, 2]).eval_uint("000"), 0)
        self.assertEqual(AerVar(AerUint(3), [1, 0, 2]).eval_uint("001"), 2)

        # overflow
        self.assertEqual(AerVar(AerUint(3), [0, 1, 2]).eval_uint("1111"), 7)
        self.assertEqual(AerVar(AerUint(5), [0, 1, 2]).eval_uint("111"), 7)

        # bool
        self.assertEqual(AerVar(AerBool(), [0, 1, 2]).eval_bool("000"), False)
        self.assertEqual(AerVar(AerBool(), [0, 1, 2]).eval_bool("001"), True)

    def test_unary_expression(self):
        """test AerUnaryExpr"""

        # !(False) = True
        self.assertEqual(AerUnaryExpr(AerUnaryOp.LogicNot, AerBoolValue(False)).eval_bool(""), True)
        # !(True) = False
        self.assertEqual(AerUnaryExpr(AerUnaryOp.LogicNot, AerBoolValue(True)).eval_bool(""), False)
        # !(!(False)) = False
        self.assertEqual(
            AerUnaryExpr(
                AerUnaryOp.LogicNot, AerUnaryExpr(AerUnaryOp.LogicNot, AerBoolValue(False))
            ).eval_bool(""),
            False,
        )
        # !(!(True)) = True
        self.assertEqual(
            AerUnaryExpr(
                AerUnaryOp.LogicNot, AerUnaryExpr(AerUnaryOp.LogicNot, AerBoolValue(True))
            ).eval_bool(""),
            True,
        )

        # !(0ul): Error
        try:
            AerUnaryExpr(AerUnaryOp.LogicNot, AerUintValue(3, 0))
            self.fail("do not reach here")
        except Exception:
            pass

        # !(1ul): Error
        try:
            AerUnaryExpr(AerUnaryOp.LogicNot, AerUintValue(3, 1))
            self.fail("do not reach here")
        except Exception:
            pass

        # ~(False): Error
        try:
            AerUnaryExpr(AerUnaryOp.BitNot, AerBoolValue(False))
            self.fail("do not reach here")
        except Exception:
            pass

        # ~(True): Error
        try:
            AerUnaryExpr(AerUnaryOp.BitNot, AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass

        # ~(0b000) = 0b111
        self.assertEqual(AerUnaryExpr(AerUnaryOp.BitNot, AerUintValue(3, 0)).eval_uint(""), 0b111)
        # ~(0b001) = 0b110
        self.assertEqual(AerUnaryExpr(AerUnaryOp.BitNot, AerUintValue(3, 1)).eval_uint(""), 0b110)
        # ~(0b00000) = 0b11111
        self.assertEqual(AerUnaryExpr(AerUnaryOp.BitNot, AerUintValue(5, 0)).eval_uint(""), 0b11111)
        # ~(0b00001) = 0b11110
        self.assertEqual(AerUnaryExpr(AerUnaryOp.BitNot, AerUintValue(5, 1)).eval_uint(""), 0b11110)
        # ~(0b10101) = 0b01010
        self.assertEqual(
            AerUnaryExpr(AerUnaryOp.BitNot, AerUintValue(5, 0b10101)).eval_uint(""), 0b01010
        )

    def test_binary_expression(self):
        """test AerBinaryExpr"""

        # (False && False) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicAnd, AerBoolValue(False), AerBoolValue(False)).eval_bool(
                ""
            ),
            False,
        )
        # (False && True) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicAnd, AerBoolValue(False), AerBoolValue(True)).eval_bool(
                ""
            ),
            False,
        )
        # (True && False) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicAnd, AerBoolValue(True), AerBoolValue(False)).eval_bool(
                ""
            ),
            False,
        )
        # (True && True) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicAnd, AerBoolValue(True), AerBoolValue(True)).eval_bool(
                ""
            ),
            True,
        )
        # (1 && 1): Error
        try:
            AerBinaryExpr(AerBinaryOp.LogicAnd, AerUintValue(3, 1), AerUintValue(3, 1))
            self.fail("do not reach here")
        except Exception:
            pass

        # (False || False) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicOr, AerBoolValue(False), AerBoolValue(False)).eval_bool(
                ""
            ),
            False,
        )
        # (False || True) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicOr, AerBoolValue(False), AerBoolValue(True)).eval_bool(
                ""
            ),
            True,
        )
        # (True || False) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicOr, AerBoolValue(True), AerBoolValue(False)).eval_bool(
                ""
            ),
            True,
        )
        # (True || True) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LogicOr, AerBoolValue(True), AerBoolValue(True)).eval_bool(
                ""
            ),
            True,
        )
        # (1 || 1): Error
        try:
            AerBinaryExpr(AerBinaryOp.LogicOr, AerUintValue(3, 1), AerUintValue(3, 1))
            self.fail("do not reach here")
        except Exception:
            pass

        # (False == False) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Equal, AerBoolValue(False), AerBoolValue(False)).eval_bool(
                ""
            ),
            True,
        )
        # (False == True) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Equal, AerBoolValue(False), AerBoolValue(True)).eval_bool(""),
            False,
        )
        # (1 == 1) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Equal, AerUintValue(3, 1), AerUintValue(3, 1)).eval_bool(""),
            True,
        )
        # (1 == 2) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Equal, AerUintValue(3, 1), AerUintValue(3, 2)).eval_bool(""),
            False,
        )

        # (False != False) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.NotEqual, AerBoolValue(False), AerBoolValue(False)).eval_bool(
                ""
            ),
            False,
        )
        # (False != True) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.NotEqual, AerBoolValue(False), AerBoolValue(True)).eval_bool(
                ""
            ),
            True,
        )
        # (1 != 1) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.NotEqual, AerUintValue(3, 1), AerUintValue(3, 1)).eval_bool(
                ""
            ),
            False,
        )
        # (1 != 2) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.NotEqual, AerUintValue(3, 1), AerUintValue(3, 2)).eval_bool(
                ""
            ),
            True,
        )
        # (False < False): error
        try:
            AerBinaryExpr(AerBinaryOp.Less, AerBoolValue(False), AerBoolValue(False))
            self.fail("do not reach here")
        except Exception:
            pass

        # (False < True): error
        try:
            AerBinaryExpr(AerBinaryOp.Less, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (1 < 1) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Less, AerUintValue(3, 1), AerUintValue(3, 1)).eval_bool(""),
            False,
        )
        # (1 < 2) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Less, AerUintValue(3, 1), AerUintValue(3, 2)).eval_bool(""),
            True,
        )

        # (False <= True): error
        try:
            AerBinaryExpr(AerBinaryOp.LessEqual, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (1 <= 1) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LessEqual, AerUintValue(3, 1), AerUintValue(3, 1)).eval_bool(
                ""
            ),
            True,
        )
        # (1 <= 2) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.LessEqual, AerUintValue(3, 1), AerUintValue(3, 2)).eval_bool(
                ""
            ),
            True,
        )

        # (False > True): error
        try:
            AerBinaryExpr(AerBinaryOp.Greater, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (1 > 1) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Greater, AerUintValue(3, 1), AerUintValue(3, 1)).eval_bool(
                ""
            ),
            False,
        )

        # (2 >= 1) = True
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.Greater, AerUintValue(3, 2), AerUintValue(3, 1)).eval_bool(
                ""
            ),
            True,
        )
        # (False >= True): error
        try:
            AerBinaryExpr(AerBinaryOp.GreaterEqual, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (1 >= 1) = True
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.GreaterEqual, AerUintValue(3, 1), AerUintValue(3, 1)
            ).eval_bool(""),
            True,
        )
        # (2 >= 1) = True
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.GreaterEqual, AerUintValue(3, 2), AerUintValue(3, 1)
            ).eval_bool(""),
            True,
        )

        # (False & True): Uint -> error
        try:
            AerBinaryExpr(AerBinaryOp.BitAnd, AerBoolValue(False), AerBoolValue(True)).eval_uint("")
            self.fail("do not reach here")
        except Exception:
            pass
        # (False & True) = False
        self.assertEqual(
            AerBinaryExpr(AerBinaryOp.BitAnd, AerBoolValue(False), AerBoolValue(True)).eval_bool(
                ""
            ),
            False,
        )
        # (0b001 & 0b001) = 0b001
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitAnd, AerUintValue(3, 0b001), AerUintValue(3, 0b001)
            ).eval_uint(""),
            0b001,
        )
        # (0b001 & 0b010) = 0b000
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitAnd, AerUintValue(3, 0b001), AerUintValue(3, 0b010)
            ).eval_uint(""),
            0b000,
        )

        # (False | True): error
        try:
            AerBinaryExpr(AerBinaryOp.BitOr, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (0b001 | 0b001) = 0b001
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(3, 0b001), AerUintValue(3, 0b001)
            ).eval_uint(""),
            0b001,
        )
        # (0b001 | 0b010) = 0b011
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(3, 0b001), AerUintValue(3, 0b010)
            ).eval_uint(""),
            0b011,
        )

        # (False ^ True): error
        try:
            AerBinaryExpr(AerBinaryOp.BitOr, AerBoolValue(False), AerBoolValue(True))
            self.fail("do not reach here")
        except Exception:
            pass
        # (0b001 ^ 0b001) = 0b000
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(3, 0b001), AerUintValue(3, 0b001)
            ).eval_uint(""),
            0b001,
        )
        # (0b001 ^ 0b010) = 0b011
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(3, 0b001), AerUintValue(3, 0b010)
            ).eval_uint(""),
            0b011,
        )

        # overflow case
        # (0b001 | 0b1010) = 0b1011
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(3, 0b001), AerUintValue(4, 0b1010)
            ).eval_uint(""),
            0b1011,
        )
        # (0b1010 | 0b001) = 0b1011
        self.assertEqual(
            AerBinaryExpr(
                AerBinaryOp.BitOr, AerUintValue(4, 0b1010), AerUintValue(3, 0b001)
            ).eval_uint(""),
            0b1011,
        )
