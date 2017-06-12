# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time
import pdb

# =============================================================================
# Extension modules
# =============================================================================
#from pyOpt import *
from pyOpt import Optimization
from pyOpt import PSQP
from pyOpt import SLSQP
from pyOpt import CONMIN
from pyOpt import COBYLA
from pyOpt import SOLVOPT
from pyOpt import KSOPT
from pyOpt import NSGA2
from pyOpt import SDPEN


# =============================================================================
# 
# =============================================================================
def objfunc(x):
    
    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = []
    
    fail = 0
    return f,g, fail
    

# =============================================================================
# 
# ============================================================================= 
opt_prob = Optimization('Rosenbrock Unconstraint Problem',objfunc)
opt_prob.addVar('x1','c',lower=-10.0,upper=10.0,value=-3.0)
opt_prob.addVar('x2','c',lower=-10.0,upper=10.0,value=-4.0)
opt_prob.addObj('f')
print opt_prob

# Instantiate Optimizer (PSQP) & Solve Problem
psqp = PSQP()
psqp.setOption('IPRINT',0)
psqp(opt_prob,sens_type='FD')
print opt_prob.solution(0)
