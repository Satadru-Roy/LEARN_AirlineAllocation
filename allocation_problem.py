import numpy as np

from openmdao.main.api import Assembly
from openmdao.lib.drivers.api import IterateUntil
from openmdao.lib.casehandlers.api import JSONCaseRecorder

from branch_and_bound.airline_subproblem import AirlineSubProblem
from branch_and_bound.branch_and_bound import BranchBoundLinear
from branch_and_bound.linear_program import LPSolver, LinProgSolver
from branch_and_bound.fleet_analysis import FleetAnalysis

class AllocationProblem(Assembly):

    def configure(self):
        self.add('iter', IterateUntil())
        self.add('airline_subproblem', AirlineSubProblem())
        self.add('branchbound_algorithm', BranchBoundLinear())
        self.add('solver', LPSolver())
        #self.add('solver', LinProgSolver())
        self.add('fleet_analysis', FleetAnalysis())

        #iteration hierachy
        self.driver.workflow.add(['airline_subproblem','iter','fleet_analysis'])
        self.iter.workflow.add(['branchbound_algorithm','solver'])

        #data connections

        # Connect Airline Allocation SubProblem Component with Branch  and Bound Algorithm Component and the solver
        self.connect('airline_subproblem.f_int',    ['branchbound_algorithm.f_int','solver.f_int'])
        self.connect('airline_subproblem.f_con',    ['branchbound_algorithm.f_con','solver.f_con'])
        self.connect('airline_subproblem.A_init',   'branchbound_algorithm.A_init')
        self.connect('airline_subproblem.b_init',   'branchbound_algorithm.b_init')
        self.connect('airline_subproblem.Aeq',      ['branchbound_algorithm.Aeq','solver.A_eq'])
        self.connect('airline_subproblem.beq',      ['branchbound_algorithm.beq','solver.b_eq'])
        self.connect('airline_subproblem.lb_init',  'branchbound_algorithm.lb_init')
        self.connect('airline_subproblem.ub_init',  'branchbound_algorithm.ub_init')

        # Connect Branch  and Bound Algorithm Component with the solver component
        self.connect('branchbound_algorithm.A', 'solver.A')
        self.connect('branchbound_algorithm.b',   'solver.b')
        self.connect('branchbound_algorithm.lb',  'solver.lb')
        self.connect('branchbound_algorithm.ub',  'solver.ub')

        # Connect solver component with the Branch  and Bound Algorithm Component (return results)
        self.connect('solver.xopt',     'branchbound_algorithm.xopt_current')
        self.connect('solver.fun_opt',     'branchbound_algorithm.relaxed_obj_current')
        self.connect('solver.exitflag_LP',    'branchbound_algorithm.exitflag_LP')

        self.connect('branchbound_algorithm.xopt', 'fleet_analysis.xopt')

        self.iter.add_stop_condition('branchbound_algorithm.exec_loop != 0')
        self.iter.max_iterations = 1000000

        #data recording
        self.recorders = [JSONCaseRecorder('airline_allocation.json')]


if __name__ == "__main__":
    ap = AllocationProblem()
    ap.branchbound_algorithm.f_int = ap.solver.f_int = ap.airline_subproblem.f_int
    ap.branchbound_algorithm.f_con = ap.solver.f_con = ap.airline_subproblem.f_con

    ap.branchbound_algorithm.A = ap.branchbound_algorithm.A_init = ap.solver.A = ap.airline_subproblem.A_init
    ap.branchbound_algorithm.b = ap.branchbound_algorithm.b_init = ap.solver.b = ap.airline_subproblem.b_init

    ap.branchbound_algorithm.Aeq = ap.solver.A_eq = ap.airline_subproblem.Aeq
    ap.branchbound_algorithm.beq = ap.solver.b_eq = ap.airline_subproblem.beq

    ap.branchbound_algorithm.lb = ap.branchbound_algorithm.lb_init = ap.solver.lb = ap.airline_subproblem.lb_init
    ap.branchbound_algorithm.ub = ap.branchbound_algorithm.ub_init = ap.solver.ub = ap.airline_subproblem.ub_init
    ap.solver.xopt = ap.branchbound_algorithm.xopt = np.zeros((ap.solver.f_int.shape[0]+ap.solver.f_con.shape[0], ))

    ap.run()

    print '\n=============================================='
    print 'Result Summary'
    print '=============================================='
    #Print Algorithm results
    print 'Exitflag status:  \t', ap.branchbound_algorithm.exitflag_BB
    print 'No. of function call: \t', ap.branchbound_algorithm.funCall
    #Print Allocation results
    print '\nAircraft Trip Details: \t\n', ap.fleet_analysis.DetailTrips
    print '\nPassenger Details: \t\n', ap.fleet_analysis.PaxDetail
    print "\nAirline's Net Profit [$]: \t", ap.fleet_analysis.Profit
    print '==============================================\n'
