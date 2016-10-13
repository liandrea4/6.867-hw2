import numpy as np
from cvxopt import matrix, solvers

def get_G_np(x, y):
  G_array = []
  for i in range(2*len(x)):
    subarray = []
    for j in range(len(y)):
      if i == j:
        subarray.append(1)
      elif i - len(x) == j:
        subarray.append(-1.)
      else:
        subarray.append(0.)
    G_array.append(subarray)
  return np.array(G_array)

def get_h_np(x, y, C):
  h_array = []
  for index in range(2 * len(x)):
    if index < len(x):
      h_array.append(float(C))
    else:
      h_array.append(0.)
  return np.array(h_array)

def solve_dual_svm_slack(x, y, C):
  P_np = np.array([float(np.dot(x[i], x[j]) * y[i] * y[j]) for i in range(len(x)) for j in range(len(x))])
  A_np = np.array([float(y_i) for y_i in y ])
  G_np = get_G_np(x, y)
  h_np = get_h_np(x, y, C)

  # define matrices
  P = matrix(P_np, (len(x), len(y)))
  q = matrix(-1., (len(x), 1))
  G = matrix(G_np, (2*len(x), len(x)))
  h = matrix(h_np, (2*len(x),1))
  A = matrix(A_np, (1, len(y)))
  b = matrix(0., (1, 1))

  # print "P: ", P
  # print "q: ", q
  # print "G: ", G
  # print "h: ", h
  # print "A: ", A
  # print "b: ", b

  # find solution
  solution = solvers.qp(P, q, G, h, A, b)
  xvals = np.array(solution['x'])
  return xvals

def get_classification_error_rate(alpha_vals, C, threshold):
  num_errors = 0
  for alpha in alpha_vals:
    if abs(alpha - C) < threshold:
      num_errors += 1
  return float(num_errors) / len(alpha_vals)




data = [
  (2,2),
  (2,3),
  (0,-1),
  (-3,-2)
]
x = [ point[0] for point in data ]
y = [ point[1] for point in data ]

# print solve_dual_svm_slack(x, y, 1)
