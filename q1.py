import numpy as np
import scipy.sparse as sp

def f_func(x, y):
    # Compute the source function
    return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

def build_poisson_matrix(N):
    # Construct the Laplacian operator matrix
    n = N - 1
    e = np.ones(n)
    T = sp.diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(n, n))
    I = sp.eye(n)
    A = sp.kron(I, T) + sp.kron(T, I)
    return -A  # Return the negative matrix

def build_rhs(N, f):
    # Construct the right-hand side vector
    n = N - 1
    h = 1.0 / N
    b = np.zeros(n * n)
    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            k = (j - 1) * n + (i - 1)
            b[k] = h**2 * f(x, y)
    return b

def reconstruct_solution(u_vec, N):
    # Reconstruct the solution vector into a grid matrix
    n = N - 1
    U = np.zeros((N+1, N+1))
    U[1:N, 1:N] = u_vec.reshape((n, n))
    return U

# Example assemble and print dimensions
if __name__ == "__main__":
    N = 32
    A = build_poisson_matrix(N)
    b = build_rhs(N, f_func)
    print("A shape:", A.shape)  
    print("b shape:", b.shape)  
