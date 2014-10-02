import numpy as np

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Array, Bool, Int

#Choose the solver that solves the relaxed programming problem

try:
    from scipy.optimize import linprog
except ImportError, e:
    print "SciPy version >= 0.15.0 is required for linprog support!!"
    pass
try:
    from lpsolve55 import *
except ImportError:
    print 'lpsolve is not available'
    pass

class LPSolver(Component):
    """ A simple component wrapper for lpsolve
    """
    #implements(ILinearProgram)

    # inputs
    f_int  = Array(iotype='in',
        desc='coefficients of the integer type design variables of the linear objective function to be maximized')

    f_con  = Array(iotype='in',
        desc='coefficients of the continuous type design variables of the linear objective function to be maximized')


    A     = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b     = Array(iotype='in',
            desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub')

    A_eq  = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    b_eq  = Array(iotype='in',
            desc='1-D array of values representing the RHS of each equality constraint (row) in A_eq')

    lb    = Array(iotype='in',
            desc='lower bounds for each independent variable in the solution')

    ub    = Array(iotype='in',
            desc='upper bounds for each independent variable in the solution')

    # outputs
    xopt     = Array(iotype='out',
            desc='independent variable vector which optimizes the linear programming problem')

    fun_opt   = Float(iotype='out',
            desc='function value')

    success = Bool(iotype='out',
              desc='flag indicating success or failure in finding an optimal solution')

    exitflag_LP  = Float(iotype='out',
              desc='exit status of the optimization: 1=optimized, -1=max iterations reached, -2=infeasible, -3=unbounded')


    def execute(self):
        """ solve the linear program """

        obj = np.hstack((self.f_int, self.f_con)).tolist()
        lp = lpsolve('make_lp', 0, len(obj))
        lpsolve('set_verbose', lp, 'IMPORTANT')
        lpsolve('set_obj_fn', lp, obj)

        i = 0
        for con in self.A:
            lpsolve('add_constraint', lp, con.tolist(), 'LE', self.b[i])
            i = i+1

        #================================================
        # Add linear equality constraints to the problem
        #================================================

        for i in range (len(self.lb)):
            lpsolve('set_lowbo', lp, i+1,  self.lb[i])
            lpsolve('set_upbo',  lp, i+1, self.ub[i])

        results = lpsolve('solve', lp)

        self.xopt   = np.array(lpsolve('get_variables', lp)[0])
        self.fun_opt = lpsolve('get_objective', lp)

        self.success = True if results == 0 else False
        if results == 0:            # optimized
            self.exitflag_LP = 1
        elif results == 2:          # infeasible
            self.exitflag_LP = -2
        elif results == 3:          # unbounded
            self.exitflag_LP = -3
        else:
            self.exitflag_LP = -1
        lpsolve('delete_lp', lp)


class LinProgSolver(Component):
    """ A simple component wrapper for lpsolve
    """
    #implements(ILinearProgram)

    # inputs
    f_int  = Array(iotype='in',
        desc='coefficients of the integer type design variables of the linear objective function to be maximized')

    f_con  = Array(iotype='in',
        desc='coefficients of the continuous type design variables of the linear objective function to be maximized')


    A     = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b     = Array(iotype='in',
            desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub')

    A_eq  = Array(iotype='in',
            desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    b_eq  = Array(iotype='in',
            desc='1-D array of values representing the RHS of each equality constraint (row) in A_eq')

    lb    = Array(iotype='in',
            desc='lower bounds for each independent variable in the solution')

    ub    = Array(iotype='in',
            desc='upper bounds for each independent variable in the solution')

    # outputs
    xopt     = Array(iotype='out',
            desc='independent variable vector which optimizes the linear programming problem')

    fun_opt   = Float(iotype='out',
            desc='function value')

    success = Bool(iotype='out',
              desc='flag indicating success or failure in finding an optimal solution')

    exitflag_LP  = Float(iotype='out',
              desc='exit status of the optimization: 1=optimized, -1=max iterations reached, -2=infeasible, -3=unbounded')

    def execute(self):
        """ solve the linear program """
        bounds = zip(self.lb.flatten(), self.ub.flatten())
        f = np.hstack((self.f_int, self.f_con))
        results = linprog(f,
                          A_eq=self.A_eq, b_eq=self.b_eq,
                          A_ub=self.A, b_ub=self.b,
                          bounds=bounds,
                          options={ 'maxiter': 1000, 'disp': True })

       
        self.success = True if results.status == 0 else False
        if self.success: 
            self.xopt = results.x
            self.fun_opt = results.fun
        else: 
            self.x = np.nan*np.ones(self.xopt.shape)
            self.fun_opt = np.nan

        # translate status to MATLAB equivalent exit flag
        if results.status == 0:         # optimized
            self.exitflag_LP = 1
        elif results.status == 1:       # max iterations
            self.exitflag_LP = 0
        elif results.status == 2:       # infeasible
            self.exitflag_LP = -2
        elif results.status == 3:       # unbounded
            self.exitflag_LP = -3
        else:
            self.exitflag_LP = -1
