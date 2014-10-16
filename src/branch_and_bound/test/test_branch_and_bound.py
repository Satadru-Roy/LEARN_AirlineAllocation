
import unittest

from branch_and_bound.airline_subproblem import AirlineSubProblem


class AirlineSubProblemTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A(self):

        asp = AirlineSubProblem()
        asp.run()
        self.assertEqual(asp.A_init.shape, (14,12))


if __name__ == "__main__":
    unittest.main()
