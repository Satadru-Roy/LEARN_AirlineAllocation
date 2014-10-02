import numpy as np
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Array

class FleetAnalysis(Component):
    """ Airline allocation result analysis compnent analyzes the allocation data and present it in the user interpretable format """

    #inputs
    xopt = Array(iotype='in',
            desc='independent variable vector which optimizes the integer programming problem')

##    fopt   = Array(iotype='in',
##            desc='optimal objective function value')

    #output
    DetailTrips = Array(iotype='out',
                        desc='details of trip performed by each aircraft type in each route')

    PaxDetail = Array(iotype='out',
                      desc = 'details of passengers carried by each aircraft type on each route')

    Profit = Float(iotype='out',
                   desc='Net profit made by the airline')

    def execute(self):

        data = self.parent.airline_subproblem.data
        J = data.inputs.DVector.shape[0]  # number of routes
        K = len(data.inputs.AvailPax)     # number of aircraft types
        KJ  = K*J

        x_hat = self.xopt[0:KJ]                # Airline allocation variable
        aa = np.where(np.abs(x_hat - 0.) < 1e-06)[0]
        x_hat[aa] = 0

        pax = self.xopt[KJ:KJ*2]             # passenger design variable
        bb = np.where(np.abs(pax - 0.) < 1e-06)[0]
        pax[bb] = 0

        RVector   = data.inputs.RVector

        detailtrips = np.zeros((K, J))
        pax_rep     = np.zeros((K, J))
        for k in range(K):
            for j in range(J):
                ind = k * J + j
                detailtrips[k, j] = 2*x_hat[ind]
                pax_rep[k, j] = 2*pax[ind]

        r, c = detailtrips.shape
        self.DetailTrips = detailtrips
        self.PaxDetail   = pax_rep

        Trips     = np.zeros((r, 1))
        FleetUsed = np.zeros((r, 1))
        Fuel      = np.zeros((r, 1))
        Tot_costnofuel       = np.zeros((r, 1))
        BlockTime = np.zeros((r, 1))
##        Nox       = np.zeros((r, 1))
        Maxpax    = np.zeros((r, 1))
        Pax       = np.zeros((r, 1))
        Miles     = np.zeros((r, 1))

        for i in range(r):
            Trips[i, 0] = np.sum(detailtrips[i, :])
            FleetUsed[i, 0] = np.ceil(np.sum(data.coefficients.BlockTime[i, :]*((1+data.constants.MH[i]))*(detailtrips[i, :]) + detailtrips[i, :]*(data.inputs.TurnAround)) / 24)
            Fuel[i, 0] = np.sum(data.coefficients.Fuelburn[i, :]*(detailtrips[i, :]))
            Tot_costnofuel[i, 0] = np.sum(data.coefficients.Tot_costnofuel[i, :]*(detailtrips[i, :]))
            BlockTime[i, 0] = np.sum(data.coefficients.BlockTime[i, :]*(detailtrips[i, :]))
##            self.Nox[i, 0] = np.sum(data.coefficients.Nox[i, :]*(detailtrips[i, :]))
            Maxpax[i, 0] = np.sum(data.inputs.AvailPax[i]*(detailtrips[i, :]))
            Pax[i, 0] = np.sum(pax_rep[i, :])
            Miles[i, 0] = np.sum(pax_rep[i, :]*(RVector.T))

        CostDetail  = data.coefficients.Tot_costnofuel*detailtrips + data.coefficients.Fuelburn*data.constants.FuelCost*detailtrips
        RevDetail   = data.inputs.TicketPrice*pax_rep
        RevArray    = np.sum(RevDetail, 0)
        CostArray   = np.sum(CostDetail, 0)
        PaxArray    = np.sum(pax_rep, 0)
        ProfitArray = RevArray - CostArray
        Revenue     = np.sum(RevDetail, axis=1)

        # record a/c performance
        PPNM        = np.zeros((1, K))
        ProfitArray = ProfitArray
        profit_v    = np.sum(ProfitArray.T)

        den_v = np.sum(PaxArray*RVector)
        PPNM  = np.array(profit_v / den_v)
        for i in range(PPNM.size-1):
            if np.isnan(PPNM[i]):
                PPNM[i] = 0

        Cost   = np.sum(Tot_costnofuel + Fuel*(data.constants.FuelCost))
        PPNM   = PPNM
        self.Profit = np.sum(RevArray - CostArray)


