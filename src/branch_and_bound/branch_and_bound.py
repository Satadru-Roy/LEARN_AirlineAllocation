from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Array, Int
import copy
import numpy as np

class Problem(object):
    """Simple container to be used by the active set tree"""
    pass


class BranchBoundLinear(Component):
    """
    Branch and Bound Component to solve linear problems
    """

    # Inputs to the Branch and Bound component
    f_int  = Array(iotype='in',
            desc='coefficients of the integer type design variables of the linear objective function to be maximized')

    f_con  = Array(iotype='in',
            desc='coefficients of the continuous type design variables of the linear objective function to be maximized')

    A_init   = Array(iotype='in',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b_init   = Array(iotype='in',
          desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A')

    Aeq = Array(iotype='in',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    beq = Array(iotype='in',
          desc='1-D array of values representing the RHS of each equality constraint (row) in Aeq')

    lb_init  = Array(iotype='in',
          desc='initial lower bounds for each independent variable in the solution')

    ub_init  = Array(iotype='in',
          desc='initial upper bounds for each independent variable in the solution')

    xopt_current = Array(iotype='in',
            desc='optimal design variable values of the relaxed solution after each iteration obtained from the solver')

    relaxed_obj_current   = Float(np.inf, iotype='in',
            desc='optimal objective function value of the relaxed solution after each iteration obtained from the solver')

    exitflag_LP  = Float(iotype='in',
              desc='exit status of the optimization: 1=optimized, -1=max iterations reached, -2=infeasible, -3=unbounded')


    # Outputs to solver that solves the relaxed problem formulation
    A = Array(iotype='out',
        desc='Updated inequality constraint coefficient matrix: [Ax<=b]')

    b = Array(iotype='out',
        desc='Updated inequality bound vector: [Ax<=b]')

    lb = Array(iotype='out',
            desc='lower bounds for each independent variable in the solution')

    ub = Array(iotype='out',
            desc='upper bounds for each independent variable in the solution')

    # Outputs to post processing component (Final output from Branch and bound algorithm)
    xopt = Array(iotype='out',
            desc='independent variable vector which optimizes the integer programming problem')

    obj_opt   = Float(iotype='out',
            desc='optimal objective function value')


    ##    x_best_relax = Array(iotype='out',
    ##            desc='independent variable vector which optimizes the relaxed programming problem')
    ##
    ##    f_best_relax   = Float(iotype='out',
    ##            desc='Optimal function value of the relaxed problem')

    exec_loop  = Float(iotype='out',
          desc='Execution loop: 0-Continue: There are active node/s present, 1-Stop:No more active node exits')

    exitflag_BB  = Int(iotype='out',
              desc='BranchBound exit flag 0-No solution foun, 1-Solution found')

    funCall  = Int(iotype='out',
              desc='number of times the solver is called')


    def __init__(self):
        super(BranchBoundLinear, self).__init__()

        self._iter = 0
        self.iter_max = 10000000
        self.funCall = 0
        self.exitflag_BB = 0
        #self.U_best = np.inf  # If you are testing the code use this line, instead of U_best = 0
        self.U_best = 0 #Do nothing: zero profit (worst case) [For airline allocation]

        self.obj_opt = 0.0
        self.can_x = []
        self.can_F = []
        self.ter_crit = 0
        self.opt_cr = 0.03
        self.strategy = 1
        self.node_num = 1
        self.tree = [1]
        self.app_cut = 0 # Cut feature currently disabled
        self.cut_num = 0
        self.exitflag_LP = 0
        self.Fsub_i = 0

        self.Aset = []


    def execute(self):
        self._iter = self._iter + 1

        self.num_int = len(self.f_int)
        self.num_des = self.num_int + len(self.f_con)

        #just make some local references
        Aset = self.Aset
        Fsub_i = self.Fsub_i

        #for the first iteration, need to put the initial problem into the active set
        if self._iter == 1:
            prob = Problem()
            prob.A    = self.A_init
            prob.b    = self.b_init
            prob.lb   = self.lb_init
            prob.ub   = self.ub_init
            prob.relaxed_obj  = 0
            prob.node = self.node_num
            prob.tree = self.tree
            prob.x_F = []
            prob.b_F = 0
            prob.eflag = 0
            Aset.append(prob)

        if self._iter > 1:
            Aset[Fsub_i].eflag = self.exitflag_LP
            Aset[Fsub_i].x_F = self.xopt_current
            Aset[Fsub_i].b_F = self.relaxed_obj_current

            if ((Aset[Fsub_i].eflag >= 1) and (Aset[Fsub_i].b_F < self.U_best)):
                # Rounding integers
                aa = np.where(np.abs(np.round(Aset[Fsub_i].x_F) - Aset[Fsub_i].x_F) <= 1e-06)
                Aset[Fsub_i].x_F[aa] = np.round(Aset[Fsub_i].x_F[aa])

                if np.linalg.norm(Aset[Fsub_i].x_F[:self.num_int] - np.round(Aset[Fsub_i].x_F[:self.num_int])) <= 1e-06:
                    print '======================='
                    print 'New solution found!'
                    print '======================='
                    self.can_x.append(Aset[Fsub_i].x_F)
                    self.can_F.append(Aset[Fsub_i].b_F)
                    self.x_best = Aset[Fsub_i].x_F.copy()
                    self.f_best = Aset[Fsub_i].b_F

                    # Discard nodes within percentage of the tolerance gap of the best feasible solution (integer)
                    # Optimal solution will be opt_cr% of the best feasible solution
                    self.U_best = self.f_best/(1+np.sign(self.f_best)*self.opt_cr)
                    del Aset[Fsub_i]  # Fathom by integrality
                    self.ter_crit = 1

                else:
                    if self.app_cut == 1:
                        A_bound = np.concatenate((np.eye(self.num_des),-1*np.eye(self.num_des)))
                        b_bound = np.concatenate((Aset[Fsub_i].ub,-1*Aset[Fsub_i].lb))

                        aa = []
                        for kk in range(len(b_bound)):
                            if b_bound[kk] != np.inf and b_bound[kk] !=0:
                                aa.append(kk)

                        A_bound = A_bound[aa][:]
                        b_bound = b_bound[aa]
                        A_mod = np.concatenate((Aset[Fsub_i].A,A_bound))
                        b_mod = np.concatenate((Aset[Fsub_i].b,b_bound))

                        A_up, b_up, cut_flag = GomoryMIR_cut(Aset[Fsub_i].x_F.flatten(), A_mod, b_mod, Aset[Fsub_i].Aeq, Aset[Fsub_i].beq, num_int)
                        Aset[Fsub_i].A = np.concatenate((Aset[Fsub_i].A, A_up))
                        Aset[Fsub_i].b = np.concatenate((Aset[Fsub_i].b, b_up))
                        if self.cut_flag == 1:
                            self.cut_num = self.cut_num + 1

                    # Branching
                    x_ind_maxfrac = np.argmax(np.abs(Aset[Fsub_i].x_F[:self.num_int] - np.round(Aset[Fsub_i].x_F[:self.num_int])))
                    x_split = Aset[Fsub_i].x_F[x_ind_maxfrac]
                    print 'Branching at node: %d at x%d = %f' % (Aset[Fsub_i].node, x_ind_maxfrac+1, x_split)
                    F_sub = [None, None]
                    for jj in 0, 1:
                        F_sub[jj] = copy.deepcopy(Aset[Fsub_i])
                        if jj == 0:
                            ub_new = np.floor(x_split)
                            if ub_new < F_sub[jj].ub[x_ind_maxfrac]:
                                F_sub[jj].ub[x_ind_maxfrac] = ub_new
                        elif jj == 1:
                            lb_new = np.ceil(x_split)
                            if lb_new > F_sub[jj].lb[x_ind_maxfrac]:
                                F_sub[jj].lb[x_ind_maxfrac] = lb_new

                        F_sub[jj].tree.append(jj+1)
                        self.node_num = self.node_num + 1
                        F_sub[jj].node = self.node_num
                    del Aset[Fsub_i]  # Fathomed by branching
                    Aset.extend(F_sub)
            else:
                del Aset[Fsub_i]  # Fathomed by infeasibility or bounds

        if not Aset: #problem stops when Aset is empty
            self.exec_loop = 1
            print '\nTerminating Branch and Bound algorithm...'
            if self.ter_crit ==1:
                self.exitflag_BB = 1
                self.xopt = self.x_best
                self.obj_opt = self.f_best
                print 'Solution found!!'
            else:
                print 'No solution found!!'
                if self._iter > self.iter_max:
                    print 'Maximum number of iterations reached!'
        else:
            # Pick A subproblem from active set
            if self.strategy == 1: # Strategy 1: Depth first search
                # Preference given to nodes with highest tree length
                max_tree = 0
                for ii in range(len(Aset)):
                    if len(Aset[ii].tree) > max_tree:
                        Fsub_i = ii
                        max_tree = len(Aset[ii].tree)
            elif self.strategy == 2: # Strategy 2: Best first search
                # Preference given to nodes with the best objective value
                Fsub = np.inf
                for ii in range(len(Aset)):
                    if Aset(ii).relaxed_obj < Fsub:
                        Fsub_i = ii
                        Fsub = Aset[ii].relaxed_obj

            #set outputs to the boundary(solver) for the next relaxed lp solve

            self.A = Aset[Fsub_i].A
            self.b = Aset[Fsub_i].b
            self.lb = Aset[Fsub_i].lb
            self.ub = Aset[Fsub_i].ub
            self.Fsub_i = Fsub_i
            self.funCall = self.funCall + 1

            if self._iter == 1:
                x_best_relax = Aset[Fsub_i].x_F
                f_best_relax = Aset[Fsub_i].b_F

            self.Aset = Aset


class BranchBoundNonLinear(Component):
    """
    Branch and Bound Component to solve linear problems
    """


    lb_init  = Array(iotype='in',
          desc='initial lower bounds for each independent variable in the solution')

    ub_init  = Array(iotype='in',
          desc='initial upper bounds for each independent variable in the solution')

    xopt_current = Array(iotype='in',
            desc='optimal design variable values of the relaxed solution after each iteration obtained from the solver')

    relaxed_obj_current  = Float(np.inf, iotype='in',
            desc='optimal objective function value of the relaxed solution after each iteration obtained from the solver')

    exitflag_NLP = Float(iotype='in',
              desc='exit status of the optimization: 1=optimized, -1=max iterations reached, -2=infeasible, -3=unbounded')


    # Outputs to solver that solves the relaxed problem formulation
    lb = Array(iotype='out',
            desc='lower bounds for each independent variable in the solution')

    ub = Array(iotype='out',
            desc='upper bounds for each independent variable in the solution')

    # Outputs to post processing component (Final output from Branch and bound algorithm)
    xopt = Array(iotype='out',
            desc='independent variable vector which optimizes the integer programming problem')

    obj_opt   = Float(iotype='out',
            desc='optimal objective function value')


    ##    x_best_relax = Array(iotype='out',
    ##            desc='independent variable vector which optimizes the relaxed programming problem')
    ##
    ##    f_best_relax   = Float(iotype='out',
    ##            desc='Optimal function value of the relaxed problem')

    exec_loop  = Float(iotype='out',
          desc='Execution loop: 0-Continue: There are active node/s present, 1-Stop:No more active node exits')

    exitflag_BB  = Int(iotype='out',
              desc='BranchBound exit flag 0-No solution foun, 1-Solution found')

    funCall  = Int(iotype='out',
              desc='number of times the solver is called')


    def __init__(self, n_int, n_cont):
        super(BranchBoundNonLinear, self).__init__()

        self._iter = 0
        self.iter_max = 10000000
        self.funCall = 0
        self.exitflag_BB = 0
        self.U_best = np.inf  # If you are testing the code use this line, instead of U_best = 0
        #self.U_best = 0 #Do nothing: zero profit (worst case) [For airline allocation]

        self.obj_opt = 0.0
        self.can_x = []
        self.can_F = []
        self.ter_crit = 0
        self.opt_cr = 0.03
        self.strategy = 1
        self.node_num = 1
        self.tree = [1]
        self.app_cut = 0 # Cut feature currently disabled
        self.cut_num = 0
        self.exitflag_NLP = 0
        self.Fsub_i = 0

        self.num_int = n_int
        self.num_des = n_cont + n_int

        self.Aset = []


    def execute(self):
        self._iter = self._iter + 1

        #just make some local references
        Aset = self.Aset
        Fsub_i = self.Fsub_i


        #for the first iteration, need to put the initial problem into the active set
        if self._iter == 1:
            prob = Problem()
            prob.lb   = self.lb_init
            prob.ub   = self.ub_init
            prob.relaxed_obj  = 0
            prob.node = self.node_num
            prob.tree = self.tree
            prob.x_F = []
            prob.b_F = 0
            prob.eflag = 0
            Aset.append(prob)




        if self._iter > 1:
            Aset[Fsub_i].eflag = self.exitflag_NLP
            Aset[Fsub_i].x_F = self.xopt_current
            Aset[Fsub_i].b_F = self.relaxed_obj_current


            if ((Aset[Fsub_i].eflag >= 1) and (Aset[Fsub_i].b_F < self.U_best)):
                # Rounding integers
                aa = np.where(np.abs(np.round(Aset[Fsub_i].x_F) - Aset[Fsub_i].x_F) <= 1e-06)
                Aset[Fsub_i].x_F[aa] = np.round(Aset[Fsub_i].x_F[aa])

                if np.linalg.norm(Aset[Fsub_i].x_F[:self.num_int] - np.round(Aset[Fsub_i].x_F[:self.num_int])) <= 1e-06:
                    print '======================='
                    print 'New solution found!'
                    print '======================='
                    self.can_x.append(Aset[Fsub_i].x_F)
                    self.can_F.append(Aset[Fsub_i].b_F)
                    self.x_best = Aset[Fsub_i].x_F.copy()
                    self.f_best = Aset[Fsub_i].b_F

                    # Discard nodes within percentage of the tolerance gap of the best feasible solution (integer)
                    # Optimal solution will be opt_cr% of the best feasible solution
                    self.U_best = self.f_best/(1+np.sign(self.f_best)*self.opt_cr)
                    del Aset[Fsub_i]  # Fathom by integrality
                    self.ter_crit = 1

                else:
                    # Branching
                    x_ind_maxfrac = np.argmax(np.abs(Aset[Fsub_i].x_F[:self.num_int] - np.round(Aset[Fsub_i].x_F[:self.num_int])))
                    x_split = Aset[Fsub_i].x_F[x_ind_maxfrac]
                    print 'Branching at node: %d at x%d = %f' % (Aset[Fsub_i].node, x_ind_maxfrac+1, x_split)
                    F_sub = [None, None]
                    for jj in 0, 1:
                        F_sub[jj] = copy.deepcopy(Aset[Fsub_i])
                        if jj == 0:
                            ub_new = np.floor(x_split)
                            if ub_new < F_sub[jj].ub[x_ind_maxfrac]:
                                F_sub[jj].ub[x_ind_maxfrac] = ub_new
                        elif jj == 1:
                            lb_new = np.ceil(x_split)
                            if lb_new > F_sub[jj].lb[x_ind_maxfrac]:
                                F_sub[jj].lb[x_ind_maxfrac] = lb_new

                        F_sub[jj].tree.append(jj+1)
                        self.node_num = self.node_num + 1
                        F_sub[jj].node = self.node_num
                    del Aset[Fsub_i]  # Fathomed by branching
                    Aset.extend(F_sub)
            else:
                del Aset[Fsub_i]  # Fathomed by infeasibility or bounds

        if not Aset: #problem stops when Aset is empty
            self.exec_loop = 1
            print '\nTerminating Branch and Bound algorithm...'
            if self.ter_crit ==1:
                self.exitflag_BB = 1
                self.xopt = self.x_best
                self.obj_opt = self.f_best
                print 'Solution found!!'
            else:
                print 'No solution found!!'
                if self._iter > self.iter_max:
                    print 'Maximum number of iterations reached!'
        else:
            # Pick A subproblem from active set
            if self.strategy == 1: # Strategy 1: Depth first search
                # Preference given to nodes with highest tree length
                max_tree = 0
                for ii in range(len(Aset)):
                    if len(Aset[ii].tree) > max_tree:
                        Fsub_i = ii
                        max_tree = len(Aset[ii].tree)
            elif self.strategy == 2: # Strategy 2: Best first search
                # Preference given to nodes with the best objective value
                Fsub = np.inf
                for ii in range(len(Aset)):
                    if Aset(ii).relaxed_obj < Fsub:
                        Fsub_i = ii
                        Fsub = Aset[ii].relaxed_obj

            #set outputs to the boundary(solver) for the next relaxed lp solve

            self.lb = Aset[Fsub_i].lb
            self.ub = Aset[Fsub_i].ub

            print "foobar", self.lb, self.ub
            self.Fsub_i = Fsub_i
            self.funCall = self.funCall + 1

            if self._iter == 1:
                x_best_relax = Aset[Fsub_i].x_F
                f_best_relax = Aset[Fsub_i].b_F



