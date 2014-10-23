import numpy as np
from numpy import zeros, array

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Enum, Float, Int, Str, Array


class NonLinearTestProblem(Component): 
    """Non linear test problem for branch and bound applications""" 

    # x1 = Float(iotype="in") 
    # x2 = Float(iotype="in") 
    x = Array([1.2,15.7], iotype="in", dtype="float")

    f = Float(iotype="out")
    g1 = Float(iotype="out")
    g2 = Float(iotype="out")

    def execute(self): 
        x1 = self.x[0]
        x2 = self.x[1]
        self.f = x1**4 + x2**2 - x1**2*x2
        self.g1 = 1 - 2./3.*x1*x2
        self.g2 = 1 + (3*x1**2 - 4*x2)/3.


