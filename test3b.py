# test_4_3_2.py

import time
import matplotlib.pyplot as plt
from mg_solver import MGSolver, f_func

# —— 小规模设置 —— 
# 这里测试 N = 16, 32, 64
Ns = [16, 32, 64]
omega, nu = 2/3, 2

times_2lvl = []
cycles_2lvl = []
times_full  = []
cycles_full  = []

for N in Ns:
    # 2-level MG (lmax=1)
    solver2 = MGSolver(N, f_func, omega, nu)
    t0 = time.time()
    d2 = solver2.solve(lmax=1, tol=1e-7)
    times_2lvl.append(time.time() - t0)
    cycles_2lvl.append(d2['cycles'])

    # full-level MG (coarsest N=8)
    solverF = MGSolver(N, f_func, omega, nu)
    t1 = time.time()
    dF = solverF.solve(lmax=solverF.level_max, tol=1e-7)
    times_full.append(time.time() - t1)
    cycles_full.append(dF['cycles'])

# Plot: Runtime vs N
plt.figure()
plt.loglog(Ns, times_2lvl, '-o', label='2-level')
plt.loglog(Ns, times_full, '-o', label='full-level')
plt.xlabel('N')
plt.ylabel('Runtime (s)')
plt.title('Runtime vs N')
plt.legend()

# Plot: Cycles vs N
plt.figure()
plt.plot(Ns, cycles_2lvl, '-o', label='2-level')
plt.plot(Ns, cycles_full, '-o', label='full-level')
plt.xlabel('N')
plt.ylabel('V-cycle count')
plt.title('V-cycles to tol vs N')
plt.legend()

print(f"{'N':^6} | {'Time (2-lvl)':^14} | {'Cycles (2-lvl)':^15} | {'Time (full)':^14} | {'Cycles (full)':^15}")
print('-'*70)
for N, t2, c2, tf, cf in zip(Ns, times_2lvl, cycles_2lvl, times_full, cycles_full):
    print(f"{N:^6} | {t2:^14.4f} | {c2:^15} | {tf:^14.4f} | {cf:^15}")

plt.show()
