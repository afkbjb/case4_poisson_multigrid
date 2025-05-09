import time
import matplotlib.pyplot as plt
from mg_solver import MGSolver, f_func

N = 128
omega, nu = 2/3, 2
solver = MGSolver(N, f_func, omega, nu)

lmax_vals = [2, 3, 4]
times    = []
coarses  = []
histories= []

for l in lmax_vals:
    t0 = time.time()
    res = solver.solve(lmax=l, tol=1e-7)
    dt = time.time() - t0

    times.append(dt)
    coarses.append(res['cycles'])
    histories.append(res['hist'])

plt.figure()
plt.plot(lmax_vals, times, '-o')
plt.xlabel('lmax')
plt.ylabel('Time (s)')
plt.title(f'Runtime vs lmax  (N={N})')

plt.figure()
plt.plot(lmax_vals, coarses, '-o')
plt.xlabel('lmax')
plt.ylabel('Number of coarse solves')
plt.title(f'Coarse solves vs lmax  (N={N})')

# Plot: Residual history
plt.figure()
for l, hist in zip(lmax_vals, histories):
    plt.semilogy(hist, label=f'lmax={l}')
plt.xlabel('Iterations (V-cycles)')
plt.ylabel('Residual norm')
plt.title(f'Residuals (N={N})')
plt.legend()

print(f"{'lmax':^6} | {'Time (s)':^10} | {'V-cycles':^10}")
print('-'*32)
for l, t, c in zip(lmax_vals, times, cycles):
    print(f"{l:^6} | {t:^10.4f} | {c:^10d}")

plt.show()
