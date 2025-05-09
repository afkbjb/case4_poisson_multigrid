import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def f_func(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def build_poisson_matrix(N):
    n = N - 1
    main = 4.0 * np.ones(n)
    off  = -1.0 * np.ones(n-1)
    T = sp.diags([off, main, off], offsets=[-1,0,1], shape=(n,n))
    I = sp.eye(n)
    A = sp.kron(I, T) + sp.kron(T, I)
    return A.tocsr()

def build_rhs(N, f):
    n = N - 1
    h2 = (1.0 / N)**2
    b = np.zeros(n*n)
    for j in range(1, N):
        y = j / N
        for i in range(1, N):
            x = i / N
            k = (j-1)*n + (i-1)
            b[k] = h2 * f(x, y)
    return b

def weighted_jacobi(A, x, b, omega, nu):
    D_inv = 1.0 / A.diagonal()
    for _ in range(nu):
        r = b - A.dot(x)
        x = x + omega * (D_inv * r)
    return x

def restrict_full_weight(r, N):
    n = N - 1
    nc = N//2 - 1
    R = r.reshape((n, n))
    Rc = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            fi, fj = 2*i+1, 2*j+1
            Rc[i,j] = (
                4*R[fi,fj]
                + 2*(R[fi-1,fj] + R[fi+1,fj] + R[fi,fj-1] + R[fi,fj+1])
                +   (R[fi-1,fj-1] + R[fi-1,fj+1]
                   +R[fi+1,fj-1] + R[fi+1,fj+1])
            ) / 16.0
    return Rc.flatten()

def prolong_bilinear(ec, N):
    n = N - 1
    nc = N//2 - 1
    E = np.zeros((n, n))
    C = ec.reshape((nc, nc))

    for i in range(nc):
        for j in range(nc):
            E[2*i+1, 2*j+1] = C[i,j]

    for i in range(1, n, 2):
        for j in range(2, n-1, 2):
            E[i,j] = 0.5 * (E[i,j-1] + E[i,j+1])

    for i in range(2, n-1, 2):
        for j in range(1, n, 2):
            E[i,j] = 0.5 * (E[i-1,j] + E[i+1,j])

    for i in range(2, n-1, 2):
        for j in range(2, n-1, 2):
            E[i,j] = 0.25 * (
                E[i-1,j-1] + E[i-1,j+1] +
                E[i+1,j-1] + E[i+1,j+1]
            )

    return E.flatten()

def v_cycle(N, x, b, omega, nu, level, level_max):
    A = build_poisson_matrix(N)

    x = weighted_jacobi(A, x, b, omega, nu)

    r = b - A.dot(x)

    if level == level_max:
        return spsolve(A, r)

    rc = restrict_full_weight(r, N)
    ec = np.zeros_like(rc)

    ec = v_cycle(N//2, ec, rc, omega, nu, level+1, level_max)

    x += prolong_bilinear(ec, N)

    x = weighted_jacobi(A, x, b, omega, nu)

    return x

def multigrid_solver(N, f, omega=2/3, nu=2, tol=1e-7, max_cycles=20):
    A = build_poisson_matrix(N)
    b = build_rhs(N, f)
    x = np.zeros_like(b)

    level_max = int(np.log2(N // 8))

    res0 = np.linalg.norm(b - A.dot(x))
    for k in range(max_cycles):
        res = np.linalg.norm(b - A.dot(x))
        print(f"Cycle {k} residual = {res:.2e}")
        if res < tol:
            print("Converged.")
            break
        x = v_cycle(N, x, b, omega, nu, 0, level_max)

    return x

if __name__ == "__main__":
    sol = multigrid_solver(64, f_func)
    print("Solution complete.")
