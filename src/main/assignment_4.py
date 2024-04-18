import numpy as np

def jacobi_method(A, b, X0, TOL, N):
    n = len(b)
    x = X0.copy()
    k = 1
    
    for k in range(N):
        x_new = [0] * n
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / A[i][i]
        
        # Check convergence
        norm_diff = sum((x_new[i] - X0[i]) ** 2 for i in range(n)) ** 0.5
        if norm_diff < TOL:
            return x_new
        
        X0 = x_new.copy()
    
    return "Maximum number of iterations exceeded"

def gauss_seidel_method(A, b, X0, TOL, N):
    n = len(b)
    x = X0.copy()
    
    for k in range(N):
        x_new = [0] * n
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / A[i][i]
        
        # Check convergence
        norm_diff = sum((x_new[i] - X0[i]) ** 2 for i in range(n)) ** 0.5
        if norm_diff < TOL:
            return x_new
        
        X0 = x_new.copy()
    
    return "Maximum number of iterations exceeded"

def sor_method(A, b, X0, omega, TOL, N):
    n = len(b)
    x = X0.copy()
    
    for k in range(N):
        x_new = [0] * n
        for i in range(n):
            x_new[i] = (1 - omega) * X0[i] + (omega / A[i][i]) * (b[i] - sum(A[i][j] * x_new[j] for j in range(n) if j != i))
        
        # Check convergence
        norm_diff = sum((x_new[i] - X0[i]) ** 2 for i in range(n)) ** 0.5
        if norm_diff < TOL:
            return x_new
        
        X0 = x_new.copy()
    
    return "Maximum number of iterations exceeded"

def iterative_refinement(A, b, N, TOL, t):
    n = len(b)
    x = np.linalg.solve(A, b)
    m = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            m[j, i] = A[j, i] / A[i, i]
    
    k = 1
    while k <= N:
        r = b - np.dot(A, x)
        y = np.linalg.solve(A, r)
        xx = x + y
        
        if k == 1:
            COND = np.linalg.norm(y, np.inf) / np.linalg.norm(x, np.inf) * 10**t
        
        if np.linalg.norm(x - xx, np.inf) < TOL:
            return xx, COND
        
        x = xx.copy()
        k += 1
    
    return "Maximum number of iterations exceeded", COND

A = [[4, 1, -1], [3, 5, 1], [2, -1, 3]]
b = [5, 7, 6]
X0 = [0, 0, 0]
TOL = 1e-6
N = 100

result = jacobi_method(A, b, X0, TOL, N)
if isinstance(result, str):
    print(result)
else:
    print("Jacobi Iterative Method Approximate solution:", result)

A = [[4, 1, -1], [3, 5, 1], [2, -1, 3]]
b = [5, 7, 6]
X0 = [0, 0, 0]
TOL = 1e-6
N = 100

result = gauss_seidel_method(A, b, X0, TOL, N)
if isinstance(result, str):
    print(result)
else:
    print("Gauss-Seidel Iterative Approximate solution:", result)

A = [[4, 1, -1], [3, 5, 1], [2, -1, 3]]
b = [5, 7, 6]
X0 = [0, 0, 0]
omega = 1.2  # Example value (choose an appropriate value)
TOL = 1e-6
N = 100

result = sor_method(A, b, X0, omega, TOL, N)
if isinstance(result, str):
    print(result)
else:
    print("SOR Method Approximate solution:", result)

A = np.array([[4, 1, -1], [3, 5, 1], [2, -1, 3]])
b = np.array([5, 7, 6])
N = 100
TOL = 1e-6
t = 6

result, cond = iterative_refinement(A, b, N, TOL, t)
if isinstance(result, str):
    print(result)
else:
    print("Iterative Refinement Method Approximate solution:", result)
    print("Approximation COND:", cond)