import time
import matplotlib.pyplot as plt
from mg_solver import MGSolver, f_func

Ns = [16, 32, 64, 128]
omega, nu = 2/3, 2

times_2lvl, cycles_2lvl, finalr_2lvl = [], [], []
times_max, cycles_max, finalr_max   = [], [], []

for N in Ns:
    # 2-level MG (lmax=1)
    solver2 = MGSolver(N, f_func, omega, nu)
    t0 = time.time()
    d2 = solver2.solve(lmax=1, tol=1e-7)
    times_2lvl.append(time.time() - t0)
    cycles_2lvl.append(d2['cycles'])
    finalr_2lvl.append(d2['hist'][-1])

    # max-level MG (coarsest N=8)
    solverF = MGSolver(N, f_func, omega, nu)
    t1 = time.time()
    dF = solverF.solve(lmax=solverF.level_max, tol=1e-7)
    times_max.append(time.time() - t1)
    cycles_max.append(dF['cycles'])
    finalr_max.append(dF['hist'][-1])

plt.figure()
plt.plot(Ns, times_2lvl, '-o', label='2-level')
plt.plot(Ns, times_max, '-o', label='max-level')
plt.xlabel('N'); plt.ylabel('Runtime (s)')
plt.title('Runtime vs N'); plt.legend()

plt.figure()
plt.plot(Ns, cycles_2lvl, '-o', label='2-level')
plt.plot(Ns, cycles_max, '-o', label='max-level')
plt.xlabel('N'); plt.ylabel('Iterations (V-cycles)')
plt.title('V-cycles to convergence'); plt.legend()

print(f"{'N':^6} | {'Time(2lvl)':^10} | {'Cycles(2lvl)':^13} | {'Res(2lvl)':^10} |"
      f" {'Time(max)':^10} | {'Cycles(max)':^13} | {'Res(max)':^10}")
print('-'*80)
for (N, t2, c2, r2, tm, cm, rm) in zip(Ns,
                                        times_2lvl, cycles_2lvl, finalr_2lvl,
                                        times_max, cycles_max, finalr_max):
    print(f"{N:^6} | {t2:^10.4f} | {c2:^13d} | {r2:^10.2e} |"
          f" {tm:^10.4f} | {cm:^13d} | {rm:^10.2e}")

plt.show()
