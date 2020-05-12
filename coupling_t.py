import argparse
from numba import cuda
import numpy as np

print(cuda.gpus)

betaeb = 0.5

##
L = 81
##

# Nx will need to be varied
Nx_list = [78, 79, 80, 81, 82]

for Nx in Nx_list: 
  print('Starting Nx' + str(Nx))
# l = 1 only for Nx = 80
  l = L/(Nx+1)

  # t = 1/2 only for Nx = 80 (l = 1)
  t = 1/(2*l**2)

  # betaeb values should be the same as before
  beta = 18
  #eB = betaeb / (((Nx / Nx_init) ** 2) * beta)
  eB = betaeb / beta



  def initialize_sines(sines):
    for n in range(1, Nx + 1):
      for x in range(1, Nx + 1):
        sines[n-1, x-1] = np.sin(np.pi * (n*x)/(Nx + 1))

  def initialize_energies(energies):
    for k in range(1, Nx + 1):
      energies[k-1] = -1 * np.cos(float(k) * np.pi / (Nx + 1)) + np.cos(np.pi / (Nx + 1))
    energies = energies*2*t

  def initialize_phi(phi):
    for n in range(Nx):
      for m in range(n+1):
        for l in range(m+1):
          phi[n, m, l] = 1.0

  def weight(n, m, l):
    if ((n != m) and (m != l) and (l != n)):
      return 6.0
    elif (n == m) and (m == l):
      return 1.0
    else:
      return 3.0



  def normalize_phi(phi):
    N = 0.

    for n in range(Nx):
      for m in range(n + 1):
        for l in range(m + 1):
          N = N + (weight(n, m, l) * (phi[n, m, l] ** 2.))

    for n in range(Nx):
      for m in range(n + 1):
        for l in range(m + 1):
          phi[n, m, l] = phi[n, m, l] / np.sqrt(N)

  def get_renormalization(phi):
    N = 0.
    for n in range(Nx):
      for m in range(n + 1):
        for l in range(m + 1):
          N = N + weight(n, m, l) * (phi[n, m, l] ** 2.)

    return np.sqrt(N)

  @cuda.jit()
  def cu_update_phi(phi, sines, energies, phi2, prefactor_num):

      (n, m, l) = cuda.grid(3)

      def weight(n, m, l):
        if ((n != m) and (m != l) and (l != n)):
          return 6.0
        elif (n == m) and (m == l):
          return 1.0
        else:
          return 3.0

      result = 0.
      if (abs(n) < phi.shape[0]) and (abs(m) < phi.shape[1]) and (abs(l) < phi.shape[2]):
          Nx = phi.shape[0]
          prefactor = prefactor_num / (energies[n] + energies[m] + energies[l] + eB)
          for np in range(Nx):
              for mp in range(np + 1):
                  for lp in range(mp + 1):
                      summand = 0.
                      for x in range(Nx):
                          summand += ((sines[n, x] * sines[np, x] * sines[m, x] * sines[mp, x] * sines[l, x] * sines[lp, x]))
                      result = result + weight(np, mp, lp) * phi[np, mp, lp] * summand
          phi2[n, m, l] = prefactor * result

  import time
  from datetime import datetime
  from pytz import timezone
  tz = timezone('EST')

  norms = []
  gs = []

  def test_g(g):
    print(f'------------ evaluating f(g) -> norm at <{g}>  ------------')
    global gs
    global norms
    gs.append(g)

    prefactor_numerator = -8.0 * g / ((Nx + 1.)**3.)

    phi = np.zeros((Nx, Nx, Nx), dtype='float32')
    phi2 = np.zeros((Nx, Nx, Nx), dtype='float32')
    sines = np.zeros((Nx, Nx), dtype='float32')
    energies = np.zeros((Nx), dtype='float32')

    initialize_sines(sines)
    initialize_energies(energies)
    initialize_phi(phi)

    norm1 = 100
    norm2 = 1
    i = 0

    while (abs(norm1 - norm2)) > 0.0001:
      loss1 = (norm1 - norm2)
      norm1 = norm2
      normalize_phi(phi)
      threadsperblock = (4, 4, 16)
      blockspergrid = (80 ,80, 80)
      start = time.time()
      cu_update_phi[blockspergrid, threadsperblock](phi, sines, energies, phi2, prefactor_numerator)
      norm2 = get_renormalization(phi2)
      end = time.time()
      phi = phi2.copy()
      loss2 = (norm1 - norm2)
      print(f'<{norm2}: completed in {end - start} seconds> ({datetime.now(tz) })')
      i+= 1
      if i > 100:
        break
    norms.append(norm2)
    print(f'------------ function evaluated in <{i}> iterations ------------')
    return 1. - norm2

  g1 = -0.001
  g2 = -4.0
  from scipy import optimize
  sol = optimize.root_scalar(test_g, bracket=[g2, g1], method='brenth', xtol=0.001)
  print(sol.root, sol.iterations, sol.function_calls)
  with open("Output2.txt", "a+") as text_file:
          text_file.write(f'{sol.root} {Nx} {betaeb} {beta}' + "\n")
