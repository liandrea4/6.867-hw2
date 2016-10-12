import numpy as np
from cvxopt import matrix, solvers

# define your matrices
# P = matrix(...)
# q = matrix(...)
# G = matrix(...)
# h = matrix(...)
# A = matrix(...)
# b = matrix(...)


# find the solution
solution = solvers.qp(P, q, G, h, A, b)
xvals = np.array(solution['x'])
