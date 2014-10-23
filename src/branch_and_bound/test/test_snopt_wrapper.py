import unittest
import numpy as np

from openmdao.main.api import Assembly, set_as_top

from branch_and_bound.simple_nonlinear import NonLinearTestProblem
from pyoptsparse_driver.pyoptsparse_driver import pyOptSparseDriver


class OptTest(Assembly): 

    def configure(self): 

        self.add('driver', pyOptSparseDriver(n_x=2))
        self.add('prob', NonLinearTestProblem())

        self.driver.add_parameter('prob.x', low=0, high=1000)
        self.driver.gradient_options.fd_form = 'central'
        self.driver.gradient_options.fd_step = 1.0e-3
        self.driver.gradient_options.fd_step_type = 'relative'

        self.driver.add_objective('prob.f')
        self.driver.add_constraint('prob.g1 < 0')
        self.driver.add_constraint('prob.g2 < 0')

class SLSQPWrapperTestCase(unittest.TestCase): 

    def test_OptBounds(self): 


        ot = set_as_top(OptTest())
        ot.driver.print_results = False
        ot.driver.lb = [0,0]
        ot.driver.ub = [1000,1000]
        ot.run()


        goal = np.array([1.0,1.502])
        error = np.abs(ot.prob.x - goal)
        self.assertTrue(np.all(error < 1e-3))
        self.assertEqual(ot.driver.exit_flag, 1)


        ot.driver.lb = [0,2]
        ot.driver.ub = [1000,1000]
        ot.run()


        goal = np.array([1.0,2.])
        error = np.abs(ot.prob.x - goal)
        self.assertTrue(np.all(error < 1e-3))
        self.assertEqual(ot.driver.exit_flag, 1)




if __name__ == "__main__": 

    unittest.main()