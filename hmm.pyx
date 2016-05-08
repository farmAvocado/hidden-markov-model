# cython: boundscheck = False
# cython: wraparound = False
# cython: nonecheck = False
# cython: cdivision = True
# cython: linetrace = True

from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX
from libc.string cimport memset, memcpy
from libc.stdio cimport printf
from libc.math cimport log as clog

DEF inf = float('inf')

cdef double forward(double[:,:] A, double[:,:] B, double[:] start, 
    int[:] O, double[:,:] alpha, double[:] scale, bint calc_likelihood=False) nogil:
  cdef:
    # n of states / observations
    int n_A = A.shape[0]
    int n_B = B.shape[1]
    int n_O = O.shape[0]
    int i, j
    double log_like = 0

  scale[0] = 0
  for i in range(n_A):
    alpha[0,i] = start[i] * B[i,O[0]]
    scale[0] += alpha[0,i]
  scale[0] = 1 / scale[0]
  for i in range(n_A):
    alpha[0,i] *= scale[0]

  for i in range(1, n_O):
    scale[i] = 0
    # could parallelize from here
    for j in range(n_A):
      alpha[i,j] = 0
      for k in range(n_A):
        alpha[i,j] += alpha[i-1,k] * A[k,j] * B[j,O[i]]
      scale[i] += alpha[i,j]
    scale[i] = 1 / scale[i]
    for j in range(n_A):
      alpha[i,j] *= scale[i]

  if calc_likelihood:
    log_like = 1.0
    for i in range(n_O):
      log_like += clog(scale[i])
    log_like = -log_like

  return log_like

cdef double backward(double[:,:] A, double[:,:] B, double[:] start,
    int[:] O, double[:,:] beta, double[:] scale, bint calc_likelihood=False) nogil:
  cdef:
    int n_A = A.shape[0]
    int n_B = B.shape[1]
    int n_O = O.shape[0]
    int i, j, k
    double likelihood = 0

  i = n_O - 1
  for j in range(n_A):
    beta[i,j] = scale[i]

  for i in range(n_O-2, -1, -1):
    for j in range(n_A):
      beta[i,j] = 0
      for k in range(n_A):
        beta[i,j] += beta[i+1,k] * A[j,k] * B[k,O[i+1]]
      beta[i,j] *= scale[i]

  if calc_likelihood:
    likelihood = 1
    for i in range(n_O):
      likelihood *= scale[i]
    likelihood = 1 / likelihood

  return likelihood

cdef double single_pass(double[:,:] A, double[:,:] B, double[:] start,
    int[:] O, double[:,:] A_cum, double[:,:] B_cum, double[:] start_cum):
  cdef:
    int n_A = A.shape[0]
    int n_B = B.shape[1]
    int n_O = O.shape[0]

    double[:] scale = <double[:n_O]>malloc(sizeof(double)*n_O)
    double[:,:] alpha = <double[:n_O,:n_A]>malloc(sizeof(double)*n_O*n_A)
    double[:,:] beta = <double[:n_O,:n_A]>malloc(sizeof(double)*n_O*n_A)
    double[:] xi_start = <double[:n_A]>malloc(sizeof(double)*n_A)
    double[:,:,:] xi = <double[:n_O-1,:n_A,:n_A]>malloc(sizeof(double)*(n_O-1)*n_A*n_A)
    double[:,:] gamma = <double[:n_O,:n_A]>malloc(sizeof(double)*n_O*n_A)
    double log_like = 0

    int i, j, k
    double z

  # E-step
  ############################################################ 
  log_like = forward(A, B, start, O, alpha, scale, True)
  backward(A, B, start, O, beta, scale)

  # calculate xi
  z = 0
  for i in range(n_A):
    xi_start[i] = start[i] * B[i,O[0]] * beta[0,i]
    z += xi_start[i]
  for i in range(n_A):
    xi_start[i] /= z              # NOTE: normalization *

  for i in range(n_O-1):
    z = 0
    for j in range(n_A):
      for k in range(n_A):
        xi[i,j,k] = alpha[i,j] * A[j,k] * B[k,O[i+1]] * beta[i+1,k]
        z += xi[i,j,k]
    for j in range(n_A):
      for k in range(n_A):
        xi[i,j,k] /= z             # NOTE: normalization *

  # calculate gamma
  for i in range(n_O):
    z = 0
    for j in range(n_A):
      gamma[i,j] = alpha[i,j] * beta[i,j]
      z += gamma[i,j]
    for j in range(n_A):
      gamma[i,j] /= z              # NOTE: normalization *

  # M-step
  ############################################################ 
  # start
  for i in range(n_A):
    start_cum[i] += xi_start[i]

  # A
  for i in range(n_A):
    for j in range(n_A):
      for k in range(n_O-1):
        A_cum[i,j] += xi[k,i,j]

  # B
  for i in range(n_A):
    for j in range(n_O):
      B_cum[i,O[j]] += gamma[j,i]

  return log_like

cpdef void baum_welch(double[:,:] A, double[:,:] B, double[:] start, list O,
    int n_iters=1, double eps=1e-12, bint verbose=False):
  cdef:
    int n_A = A.shape[0]
    int n_B = B.shape[1]

    int i, j
    double z
    int it
    int[:] o

    double[:,:] A_cum = <double[:n_A,:n_A]>malloc(sizeof(double)*n_A*n_A)
    double[:,:] B_cum = <double[:n_A,:n_B]>malloc(sizeof(double)*n_A*n_B)
    double[:] start_cum = <double[:n_A]>malloc(sizeof(double)*n_A)
    double plog_like = -inf, log_like, delta = 0

  for it in range(n_iters):
    memset(&A_cum[0,0], 0, sizeof(double)*n_A*n_A)
    memset(&B_cum[0,0], 0, sizeof(double)*n_A*n_B)
    memset(&start_cum[0], 0, sizeof(double)*n_A)
    log_like = 0
    for o in O:
      log_like += single_pass(A, B, start, o, A_cum, B_cum, start_cum)

    delta = log_like - plog_like
    plog_like = log_like
    if verbose:
      printf('iter %d, improvement %e\n', it, delta)
    if delta <= eps:
      break

    z = 0
    for i in range(n_A):
      z += start_cum[i]
    for i in range(n_A):
      start[i] = start_cum[i] / z

    for i in range(n_A):
      z = 0
      for j in range(n_A):
        z += A_cum[i,j]
      for j in range(n_A):
        A[i,j] = A_cum[i,j] / z

      z = 0
      for j in range(n_B):
        z += B_cum[i,j]
      for j in range(n_B):
        B[i,j] = B_cum[i,j] / z
