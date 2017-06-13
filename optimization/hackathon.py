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
#List of references in the queue with their priority order
opt_prob.addVar('x1','c',lower=-10.0,upper=10.0,value=-3.0)
#The Processing capacity
opt_prob.addVar('x2','c',lower=-10.0,upper=10.0,value=-4.0)
opt_prob.addObj('f')
#Constraints
#Each queue has a given processing capacity expressed in number of workers
#Each worker can process 35 number of references per week
#Notice period to change the number of workers for one step is 8 weeks
#Max total capacity increase is 20% of nominal capacity
#Fast-lange capacity is fixed equal to 10% planned capacity
#Cost to be added in objective function for revising one date
# is equal to 1000 * number of overdue days

print opt_prob

# Instantiate Optimizer (PSQP) & Solve Problem
psqp = PSQP()
psqp.setOption('IPRINT',0)
psqp(opt_prob,sens_type='FD')
print opt_prob.solution(0)
