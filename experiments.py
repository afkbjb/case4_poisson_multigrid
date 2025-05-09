# mg_experiments_4_3.py

import time
import numpy as np
import matplotlib.pyplot as plt
from mg_solver import MGSolver, f_func

# Multigrid 参数
omega, nu = 2/3, 2

N = 128
solver128 = MGSolver(N, f_func, omega, nu)
lmax_vals = list(range(2, solver128.level_max + 1))

runtimes1    = []
coarse_solves1 = []
histories1   = []

for l in lmax_vals:
    start = time.time()
    result = solver128.solve(lmax=l, tol=1e-7)
    dt = time.time() - start

    runtimes1.append(dt)
    coarse_solves1.append(result['coarse'])
    histories1.append(result['hist'])

# Plot 1: Runtime vs lmax
plt.figure()
plt.plot(lmax_vals, runtimes1, '-o')
plt.xlabel('lmax')
plt.ylabel('Runtime (s)')
plt.title(f'Runtime vs lmax (N={N})')
plt.grid(True)

# Plot 2: Coarse‐solves vs lmax
plt.figure()
plt.plot(lmax_vals, coarse_solves1, '-o')
plt.xlabel('lmax')
plt.ylabel('Number of coarse solves')
plt.title(f'Coarse solves vs lmax (N={N})')
plt.grid(True)

# Plot 3: Residual history
plt.figure()
for l, hist in zip(lmax_vals, histories1):
    plt.semilogy(hist, label=f'lmax={l}')
plt.xlabel('V-cycle #')
plt.ylabel('Residual norm')
plt.title(f'Residual history (N={N})')
plt.legend()
plt.grid(True)

# 2) N = [16,32,64,128,256] 对比 2-level vs full-level
Ns = [16, 32, 64, 128]
rtime_two, rtime_full = [], []
cycles_two, cycles_full = [], []

for N in Ns:
    print(f"Running N={N} ")
    # 2-layer MG: lmax=1
    solver2 = MGSolver(N, f_func, omega, nu)
    t0 = time.time()
    d2 = solver2.solve(lmax=1, tol=1e-7)
    t2 = time.time() - t0

    # full-level MG: lmax = log2(N/4)
    solverF = MGSolver(N, f_func, omega, nu)
    t0 = time.time()
    dF = solverF.solve(lmax=solverF.level_max, tol=1e-7)
    tF = time.time() - t0

    rtime_two.append(t2)
    cycles_two.append(d2['cycles'])
    rtime_full.append(tF)
    cycles_full.append(dF['cycles'])

# Plot 4: Runtime vs N (log-log)
plt.figure()
plt.loglog(Ns, rtime_two, '-o', label='2-level')
plt.loglog(Ns, rtime_full, '-o', label='full-level')
plt.xlabel('N')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs N')
plt.legend()
plt.grid(True, which='both')

# Plot 5: V-cycle count vs N
plt.figure()
plt.plot(Ns, cycles_two, '-o', label='2-level')
plt.plot(Ns, cycles_full, '-o', label='full-level')
plt.xlabel('N')
plt.ylabel('Number of V-cycles')
plt.title('Cycles to tol vs N')
plt.legend()
plt.grid(True)

plt.show()
