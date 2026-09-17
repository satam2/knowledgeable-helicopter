import unittest
import numpy as np
import simplex


class SimplexContracts(unittest.TestCase):
    def test_known_convex_optimum_and_unknown_airport(self):
        first = np.array([-100., 0., 100., 10000.])
        second = first + np.array([20., 60., -90., 20000.])
        y = first + .3 * (second - first)
        model = simplex.solve(y, first, second, ["A", "A", "B", "B"])
        self.assertAlmostEqual(model["global_alpha"], .3)
        np.testing.assert_allclose(simplex.predict(model, first, second, ["NEW"] * 4, True), y)

    def test_shrinkage_matches_closed_form(self):
        first = np.zeros(4)
        second = np.ones(4)
        y = np.array([0., 0., 1., 1.])
        model = simplex.solve(y, first, second, ["A", "A", "B", "B"])
        self.assertEqual(model["global_alpha"], .5)
        self.assertAlmostEqual(model["airports"]["A"]["alpha"], 500 / 1002)
        self.assertAlmostEqual(model["airports"]["B"]["alpha"], 502 / 1002)

    def test_unmodified_extreme_label_changes_optimum_and_bounds_hold(self):
        first = np.zeros(4)
        second = np.array([1., 2., 3., 100000.])
        model = simplex.solve(second, first, second, ["A"] * 4)
        self.assertEqual(model["global_alpha"], 1.)
        negative = simplex.solve(-second, first, second, ["A"] * 4)
        self.assertEqual(negative["global_alpha"], 0.)
        np.testing.assert_array_equal(simplex.predict(model, first, second, ["A"] * 4, True), second)

    def test_identical_components_and_schema_errors(self):
        values = np.array([1., 2., 3.])
        model = simplex.solve(values, values, values, ["A"] * 3)
        np.testing.assert_array_equal(simplex.predict(model, values, values, ["NEW"] * 3, True), values)
        with self.assertRaises(ValueError):
            simplex.solve(values, values[:2], values, ["A"] * 3)
        with self.assertRaises(ValueError):
            simplex.solve([np.nan], [1.], [2.], ["A"])


if __name__ == "__main__":
    unittest.main()
