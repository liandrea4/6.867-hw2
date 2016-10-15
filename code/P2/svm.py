import numpy     as np
from cvxopt      import matrix, solvers
import math

###### SVM with slack ######

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

  solution = solvers.qp(P, q, G, h, A, b)
  xvals = np.array(solution['x'])
  return xvals



###### SVM with kernels ######

def linear_kernel_fn(x_i, x_j):
  return np.dot(x_i, x_j)

def make_gaussian_rbf_kernel_fn(gamma):
  def gaussian_rbf_kernel_fn(x_i, x_j):
    magnitude = np.linalg.norm(x_i - x_j) ** 2
    return math.exp(-1 * gamma * magnitude)
  return gaussian_rbf_kernel_fn


def solve_dual_svm_kernel(x, y, C, kernel_fn, K_hat_matrix=None):
  if K_hat_matrix is not None:
    P_np = K_hat_matrix
  else:
    P_np = np.array([float(kernel_fn(x[i], x[j]) * y[i] * y[j]) for i in range(len(x)) for j in range(len(x))])

  A_np = np.array([float(y_i) for y_i in y ])
  G_np = get_G_np(x, y)
  h_np = get_h_np(x, y, C)

  P = matrix(P_np, (len(x), len(y)))
  q = matrix(-1., (len(x), 1))
  G = matrix(G_np, (2*len(x), len(x)))
  h = matrix(h_np, (2*len(x),1))
  A = matrix(A_np, (1, len(y)))
  b = matrix(0., (1, 1))

  solution = solvers.qp(P, q, G, h, A, b)
  xvals = np.array(solution['x'])
  return xvals


if __name__ == '__main__':
  C = 1.

  ###### Dual form SVM with slack ######

  data = [
    (2,2),
    (2,3),
    (0,-1),
    (-3,-2)
  ]
  x = [ point[0] for point in data ]
  y = [ point[1] for point in data ]

  # print solve_dual_svm_slack(x, y, C)

  ###### Dual form SVM with kernel ######
  gamma = 1.

  radial_basis_fn = make_gaussian_rbf_kernel_fn(gamma)
  print solve_dual_svm_kernel(x, y, C, radial_basis_fn)



