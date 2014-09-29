"""
    OptimizationFiles.py
    The file contains objective and constraint functions required for the airline allocation.

    Original MATLAB code developed by: Satadru Roy*

    MATLAB code converted to Python by: OpenMDAO Team

    Last modified by: Satadru Roy*

    *School of Aeronautics and Astronautics
    Purdue University
"""

import numpy as np

class TwoAircraftThreeRoute(object):
    """baseline data for a 2 aircraft, 3 route sub-problem"""
    def __init__(self):
      self.coefficients = object # the fuelburn, Doc and Block time for each aircraft type on each route
      self.constants = object # fuel cost and the man-maintenance hours required per block hours of service
      self.inputs = object # Inputs to the airline allocation. AvailPax factors in the 80% load factor
      self.outputs = object # Ticket Price generated from FLEET revenue model

      self.coefficients.Fuelburn = np.array([[24794.94, 18617.62, 12722.44],[25774.5, 19431.4, 13370.31]])
      self.coefficients.Tot_costnofuel = np.array([[24050.53, 18864.80, 13686.23],[29528.54, 23558.28, 17544.29]])
      self.coefficients.BlockTime = np.array([[4.8891, 3.7773, 2.6617],[4.8491, 3.7473, 2.6417]])

      self.constants.FuelCost = 0.2431
      self.constants.MH = np.array([[0.936],[0.948]])

      self.inputs.TurnAround = 1.0
      self.inputs.AvailPax = np.array([[107.0],[122.0]])
      self.inputs.ACNum = np.array([[6],[4]])
      self.inputs.RVector = np.array([[1999.6], [1498.8], [1000.8]])
      self.inputs.DVector = np.array([[1, 300.0],[2, 700.0],[3, 220.0]])
      self.inputs.Year = 2014

      self.inputs.MaxTrip = np.zeros((K*J,1))

      K = len(ACNum)
      J = len(RVector)
      rw = 0
      for kk in range(K):
          for jj in range(J):
              self.inputs.MaxTrip[rw,0] = self.ACNum[kk,0]*np.ceil(12/(self.coefficients.BlockTime[kk,jj]*(1+self.constants.MH[kk,0]) + self.TurnAround))
              rw += 1

      self.outputs.TicketPrice = np.array([[295.8682, 235.3350, 176.7478],[308.7701, 248.2936, 188.5960]])



class AirlineAllocationProblem(Component):

    # inputs
    # No inputs required. Data directly read from the MainConstant file.
    # In the future release, the input file (MainConstant needs to be supplied
    # as an input connection to the component

    # outputs
    f_int  = Array(iotype='in',
            desc='coefficients of the integer type design variables of the linear objective function to be maximized')

    f_con  = Array(iotype='in',
            desc='coefficients of the continuous type design variables of the linear objective function to be maximized')

    A   = Array(iotype='out',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x')

    b   = Array(iotype='out',
          desc='1-D array of values representing the upper-bound of each inequality constraint (row) in A')

    Aeq = Array(iotype='out',
          desc='2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x')

    beq = Array(iotype='out',
          desc='1-D array of values representing the RHS of each equality constraint (row) in Aeq')

    lb  = Array(iotype='out',
          desc='lower bounds for each independent variable in the solution')

    ub  = Array(iotype='out',
          desc='upper bounds for each independent variable in the solution')


    def __init__(self, data=None):
      if data is None:
        data = TwoAircraftThreeRoute()
        
      self.data = data


    #=========================================
    #Airline allocation objective function
    #=========================================
    def get_objective(self):
        """ Generate the objective fucntion coefficients for linprog
            Returns the coefficients for the integer and continuous type design variables
        """

        data = self.data

        J = data.inputs.DVector.shape[0]  # number of routes
        K = len(data.inputs.AvailPax)     # number of aircraft types
        KJ = K*J

        fuelburn  = data.coefficients.Fuelburn
        Tot_costnofuel = data.coefficients.Tot_costnofuel
        price     = data.outputs.TicketPrice
        fuelcost  = data.constants.FuelCost

        obj_int = np.zeros((KJ, 1))
        obj_con = np.zeros((KJ, 1))

        for kk in xrange(K):
            for jj in xrange(J):
                col = kk*J + jj
                obj_int[col] = Tot_costnofuel[kk, jj] + fuelcost * fuelburn[kk, jj]
                obj_con[col] = -price[kk, jj]

        return obj_int.flatten(), obj_con.flatten()

    #=========================================
    #Airline allocation constraint function
    #=========================================
    def get_constraints(self):
        """ Generate the constraint matrix/vector for linprog
        """

        data = self.data

        J = data.inputs.DVector.shape[0]  # number of routes
        K = len(data.inputs.AvailPax)     # number of aircraft types
        KJ  = K*J
        KJ2 = KJ*2

        dem   = data.inputs.DVector[:, 1].reshape(-1, 1)
        BH    = data.coefficients.BlockTime
        MH    = data.constants.MH.reshape(-1, 1)
        cap   = data.inputs.AvailPax.flatten()
        fleet = data.inputs.ACNum.reshape(-1, 1)
        t     = data.inputs.TurnAround

        # Upper demand constraint
        A1 = np.zeros((J, KJ2))
        b1 = dem
        for jj in xrange(J):
            for kk in xrange(K):
                col = K*J + kk*J + jj
                A1[jj, col] = 1

        # Lower demand constraint
        A2 = np.zeros((J, KJ2))
        b2 = -0.2 * dem
        for jj in xrange(J):
            for kk in xrange(K):
                col = K*J + kk*J + jj
                A2[jj, col] = -1

        # Aircraft utilization constraint
        A3 = np.zeros((K, KJ2))
        b3 = np.zeros((K, 1))
        for kk in xrange(K):
            for jj in xrange(J):
                col = kk*J + jj
                A3[kk, col] = BH[kk, jj]*(1 + MH[kk, 0]) + t
            b3[kk, 0] = 12*fleet[kk]

        # Aircraft capacity constraint
        A4 = np.zeros((KJ, KJ2))
        b4 = np.zeros((KJ, 1))
        rw = 0
        for kk in xrange(K):
            for jj in xrange(J):
                col1 = kk*J + jj
                A4[rw, col1] = 0.-cap[kk]
                col2 = K*J + kk*J + jj
                A4[rw, col2] = 1
                rw = rw + 1

        A = np.concatenate((A1, A2, A3, A4))
        b = np.concatenate((b1, b2, b3, b4))
        return A, b

    def execute(self):

        # linear objective coefficients
        objective = self.get_objective(data)
        self.f_int = objective[0]                # integer type design variables
        self.f_con = objective[1]                # continuous type design variables

        # coefficient matrix for linear inequality constraints, Ax <= b
        constraints = self.get_constraints(data)
        self.A = constraints[0]
        self.b = constraints[1]

        # coefficient matrix for linear equality constraints, Aeqx <= beq (N/A)
        self.Aeq = np.ndarray(shape=(0, 0))
        self.beq = np.ndarray(shape=(0, 0))

        J = data.inputs.DVector.shape[0]    # number of routes
        K = len(data.inputs.AvailPax)       # number of aircraft types

        # lower and upper bounds
        self.lb = np.zeros((2*K*J, 1))
        self.ub = np.concatenate((
            np.ones((K*J, 1)) * data.inputs.MaxTrip.reshape(-1, 1),
            np.ones((K*J, 1)) * np.inf
        ))

    def get_objective(self, data):
        """ generate the objective matrix for linprog
            returns the coefficients for the integer and continuous design variables
        """
        # sharing function definition with non-openmdao code
        return get_objective(data)

    def get_constraints(self, data):
        """ generate the constraint matrix/vector for linprog
        """
        # sharing function definition with non-openmdao code
        return get_constraints(data)
