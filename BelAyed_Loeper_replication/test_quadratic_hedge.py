"""Mathematical invariants and independent KKT verification."""

import unittest
import numpy as np
from quadratic_hedge import (
    solve_quadratic_trade,
    minimum_change_quadratic,
    InfeasibleHedgeError,
)


class QuadraticHedgeTests(unittest.TestCase):
    """Regression checks for the quadratic hedge and its rebalancing threshold."""

    def setUp(self):
        """Initialize a reproducible random generator for each test."""
        self.rng = np.random.default_rng(4550643)

    def test_identity_equals_moore_penrose(self):
        """Check the identity-metric trade against the Moore–Penrose solution."""
        M = self.rng.normal(size=(100, 3, 14))
        r = self.rng.normal(size=(100, 3))
        dw, info = solve_quadratic_trade(M, r)
        ref = np.einsum("nji,ni->nj", np.linalg.pinv(M), r)
        np.testing.assert_allclose(dw, ref, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(info["constraint_residual"], 0, atol=2e-14)

    def test_general_Q_against_independent_KKT(self):
        """Compare a nonidentity metric against an independent KKT solve."""
        M = self.rng.normal(size=(3, 8))
        C = self.rng.normal(size=(8, 8))
        Q = C @ C.T + np.eye(8)
        r = self.rng.normal(size=3)
        dw, _ = solve_quadratic_trade(M, r, Q)
        K = np.block([[Q, M.T], [M, np.zeros((3, 3))]])
        solution = np.linalg.solve(K, np.r_[np.zeros(8), r])
        np.testing.assert_allclose(dw, solution[:8], rtol=2e-13, atol=2e-14)
        # Every nullspace perturbation preserves the constraint and raises cost.
        null = np.linalg.svd(M, full_matrices=True)[2][3:].T
        for _ in range(10):
            trial = dw + null @ self.rng.normal(size=5)
            self.assertGreater(trial @ Q @ trial, dw @ Q @ dw)

    def test_zero_residual_unique_zero(self):
        """Check that zero residual gives a zero trade."""
        M = self.rng.normal(size=(3, 8))
        dw, _ = solve_quadratic_trade(M, np.zeros(3))
        np.testing.assert_array_equal(dw, np.zeros(8))

    def test_nonzero_residual_nonzero_trade(self):
        """Check that matching a nonzero feasible residual requires a trade."""
        dw, _ = solve_quadratic_trade(np.array([[1.0, 2.0]]), np.array([0.01]))
        self.assertGreater(np.linalg.norm(dw), 0)

    def test_feasible_dependent_rows(self):
        """Check feasible constraints with linearly dependent sensitivity rows."""
        M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        dw, info = solve_quadratic_trade(M, np.array([1.0, 2.0]))
        self.assertEqual(info["rank"], 1)
        np.testing.assert_allclose(M @ dw, [1, 2], atol=1e-14)

    def test_infeasible_constraints_rejected(self):
        """Check that contradictory constraints raise an error."""
        with self.assertRaises(InfeasibleHedgeError):
            solve_quadratic_trade(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 3.0]))

    def test_non_spd_Q_rejected(self):
        """Check that an invalid quadratic metric is rejected."""
        with self.assertRaises(np.linalg.LinAlgError):
            solve_quadratic_trade(np.array([[1.0, 2.0]]), np.array([1.0]), np.diag([1.0, -1.0]))

    def test_threshold_has_explicit_nonzero_no_trade_band(self):
        """Check the residual threshold for trading versus retaining the previous hedge."""
        M = np.array([[[1.0, 2.0]]])
        b = np.array([[5e-6]])
        previous = np.zeros((1, 2))
        _, dw, do = minimum_change_quadratic(M, b, previous, tolerance=1e-5)
        self.assertFalse(do[0])
        np.testing.assert_array_equal(dw, 0)
        _, dw, do = minimum_change_quadratic(M, b, previous, tolerance=0)
        self.assertTrue(do[0])
        np.testing.assert_allclose(M[0] @ dw[0], b[0])

    def test_self_similar_row_rescaling(self):
        """Check trade invariance under consistent row and target rescaling."""
        M = self.rng.normal(size=(3, 8))
        r = self.rng.normal(size=3)
        scales = np.array([1.0, np.exp(0.5 * 2), np.exp(2) * (0.8) ** 0.01])
        x, _ = solve_quadratic_trade(M, r)
        xx, _ = solve_quadratic_trade(scales[:, None] * M, scales * r)
        np.testing.assert_allclose(x, xx, atol=1e-13)
