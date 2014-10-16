import numpy as np

from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import IterateUntil
from openmdao.lib.casehandlers.api import JSONCaseRecorder

from branch_and_bound.branch_and_bound_nonlinear import BranchBoundNonLinear
from branch_and_bound.nonlinear import NonLinearTestProblem, BandBSLSQPdriver

class NonLinTest(Assembly):

    def configure(self):
        self.add('driver', IterateUntil())
        self.add('branchbound_algorithm', BranchBoundNonLinear(n_int = 2, n_contin = 0))
        self.add('nonlinopt', BandBSLSQPdriver())
        self.add('nonlin_test_prob', NonLinearTestProblem())

        #nonlin problem formulation
        self.nonlinopt.add_parameter('nonlin_test_prob.x', low=0, high=1e15)

        self.nonlinopt.add_objective('nonlin_test_prob.f')
        self.nonlinopt.add_constraint('nonlin_test_prob.g1 < 0')
        self.nonlinopt.add_constraint('nonlin_test_prob.g2 < 0')

        #iteration hierachy
        self.driver.workflow.add(['branchbound_algorithm','nonlinopt'])
        self.nonlinopt.workflow.add('nonlin_test_prob')

        #data connections
        # Connect solver component with the Branch  and Bound Algorithm Component (return results)
        self.connect('nonlin_test_prob.x',     'branchbound_algorithm.xopt_current')
        self.connect('nonlin_test_prob.f',     'branchbound_algorithm.relaxed_obj_current')
        self.connect('nonlinopt.exit_flag',    'branchbound_algorithm.exitflag_NLP')


        # Connect Airline Allocation SubProblem Component with Branch  and Bound Algorithm Component and the solver
        # Connect Branch  and Bound Algorithm Component with the solver component
        self.connect('branchbound_algorithm.lb', 'nonlinopt.lb')
        self.connect('branchbound_algorithm.ub', 'nonlinopt.ub')

        self.driver.add_stop_condition('branchbound_algorithm.exec_loop != 0')
        self.driver.max_iterations = 1000000



if __name__ == "__main__":
    nlt = NonLinTest()

    nlt.nonlinopt.lb = nlt.branchbound_algorithm.lb_init = [0.,0.]
    nlt.nonlinopt.ub = nlt.branchbound_algorithm.ub_init = [1e15, 1e15]

    nlt.run()

    print "x_opt: ", nlt.branchbound_algorithm.xopt
    print "obj_opt: ", nlt.branchbound_algorithm.obj_opt

    # from openmdao.util.dotgraph import plot_graph
    # plot_graph(nlt._reduced_graph)






