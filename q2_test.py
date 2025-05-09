import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def f_func(x, y):
    return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

def build_poisson_matrix(N):
    n = N - 1
    main = 4.0 * np.ones(n)
    off  = -1.0 * np.ones(n-1)
    T = sp.diags([off, main, off], offsets=[-1,0,1], shape=(n,n))
    I = sp.eye(n)
    A = sp.kron(I, T) + sp.kron(T, I)
    return -A.tocsr()

def build_rhs(N, f):
    n = N - 1
    h2 = (1.0/N)**2
    b = np.zeros(n*n)
    for j in range(1, N):
        y = j / N
        for i in range(1, N):
            x = i / N
            k = (j-1)*n + (i-1)
            b[k] = h2 * f(x, y)
    return b

def weighted_jacobi(A, x, b, omega, nu):
    # Weighted Jacobi smoother
    omega = float(omega)
    nu = int(nu)
    assert 0 < omega < 1, "omega must be in (0,1)"
    assert nu > 0, "nu must be positive"
    D_inv = 1.0 / A.diagonal()
    for _ in range(nu):
        r = b - A.dot(x)
        x = x + omega * (D_inv * r)
    return x

def restrict_full_weight(r, N):
    # Full-weighted restriction operator from fine grid to coarse grid
    n = N - 1; nc = N//2 - 1
    R = r.reshape((n, n))
    Rc = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            fi, fj = 2*i+1, 2*j+1
            Rc[i,j] = (
                4*R[fi,fj] 
                + 2*(R[fi-1,fj] + R[fi+1,fj] + R[fi,fj-1] + R[fi,fj+1])
                + (R[fi-1,fj-1] + R[fi-1,fj+1] + R[fi+1,fj-1] + R[fi+1,fj+1])
            ) / 16.0
    return Rc.flatten()

def prolong_bilinear(ec, N):
    # Bilinear interpolation operator from coarse grid to fine grid
    n = N - 1; nc = N//2 - 1
    E = np.zeros((n,n))
    C = ec.reshape((nc,nc))
    # Injection
    for i in range(nc):
        for j in range(nc):
            E[2*i+1,2*j+1] = C[i,j]
    # Horizontal interpolation
    for i in range(1,n,2):
        for j in range(2,n-1,2):
            E[i,j] = 0.5*(E[i,j-1] + E[i,j+1])
    # Vertical interpolation
    for i in range(2,n-1,2):
        for j in range(1,n,2):
            E[i,j] = 0.5*(E[i-1,j] + E[i+1,j])
    # Diagonal interpolation
    for i in range(2,n-1,2):
        for j in range(2,n-1,2):
            E[i,j] = 0.25*(E[i-1,j-1] + E[i-1,j+1] + E[i+1,j-1] + E[i+1,j+1])
    return E.flatten()

def v_cycle(N, x, b, omega, nu, level, level_max):
    # Recursive V-cycle algorithm
    A = build_poisson_matrix(N)
    x = weighted_jacobi(A, x, b, omega, nu)
    r = b - A.dot(x)
    if level == level_max:
        x = spsolve(A, b)
    else:
        bc = restrict_full_weight(r, N)
        xc = np.zeros_like(bc)
        xc = v_cycle(N//2, xc, bc, omega, nu, level+1, level_max)
        x += prolong_bilinear(xc, N)
        x = weighted_jacobi(A, x, b, omega, nu)
    return x

def multigrid_solver(N, f, omega=2/3, nu=2, tol=1e-8, max_cycles=20):
    # Multigrid solver
    omega = float(omega); nu = int(nu)
    assert 0 < omega < 1, "omega must be in (0,1)"
    assert nu > 0 and nu < 20, "nu should be reasonable"
    A = build_poisson_matrix(N)
    b = build_rhs(N, f)
    x = np.zeros_like(b)
    level_max = int(np.log2(N//4))
    prev_res = np.linalg.norm(b - A.dot(x))
    for cycle in range(1, max_cycles+1):
        res0 = np.linalg.norm(b - A.dot(x))
        print(f"Cycle {cycle-1} start residual: {res0:.2e}")
        if res0 < tol:
            print("Converged!")
            break
        if res0 > prev_res * 1.05:
            print("Divergence detected; stopping.")
            break
        x = v_cycle(N, x, b, omega, nu, level=0, level_max=level_max)
        res1 = np.linalg.norm(b - A.dot(x))
        rate = res1/res0
        print(f"Cycle {cycle} end   residual: {res1:.2e}, rate: {rate:.2f}")
        if rate > 0.7:
            print("Convergence too slow; consider adjusting parameters.")
            break
        prev_res = res0
    return x

if __name__ == "__main__":
    sol = multigrid_solver(64, f_func)
    print("Solution complete.")
