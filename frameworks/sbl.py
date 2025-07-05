import numpy as np

def sbl(theta, y, noise_var, max_iter=500, stopping_criterion=1e-4):
    """
    Sparse Bayesian Learning (SBL) for Multiple Measurement Vectors (MMV).

    Returns:
    - gamma_new: Final soft gamma vector
    - mu_x: Final posterior mean estimate
    - gamma_history: List of gamma vectors (length = max_iter)
    - mu_x_history: List of mu_x matrices (length = max_iter)
    - iteration_count: Number of iterations executed before convergence
    """

    M = y.shape[1]
    L = theta.shape[0]
    N = theta.shape[1]

    # Initialize Gamma (diagonal covariance)
    Gamma = np.eye(N) * 0.1

    # History tracking
    gamma_history = []
    mu_x_history = []
    iteration_count = max_iter

    for t in range(max_iter):
        # E-step
        Sigma_y = theta @ Gamma @ theta.conj().T + noise_var * np.eye(L)
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_x = Gamma - Gamma @ theta.conj().T @ Sigma_y_inv @ theta @ Gamma
        mu_x = Gamma @ theta.conj().T @ Sigma_y_inv @ y

        # Gamma update
        gamma_new = (np.linalg.norm(mu_x, axis=1) ** 2) / M + np.real(np.diag(Sigma_x))
        gamma_new = np.maximum(gamma_new, 1e-8)  # Ensure positivity

        # Save history
        gamma_history.append(gamma_new.copy())
        mu_x_history.append(mu_x.copy())

        # Convergence check
        gamma_old = np.diag(Gamma)
        if np.linalg.norm(gamma_new - gamma_old) < stopping_criterion:
            iteration_count = t + 1
            print(f"Converged after {iteration_count} iterations")
            break

        # Update Gamma
        Gamma = np.diagflat(gamma_new)

    # Pad history if converged early
    pad_len = max_iter - len(gamma_history)
    if pad_len > 0:
        gamma_pad = [gamma_history[-1].copy()] * pad_len
        mu_x_pad = [mu_x_history[-1].copy()] * pad_len
        gamma_history.extend(gamma_pad)
        mu_x_history.extend(mu_x_pad)

    return gamma_new, mu_x, gamma_history, mu_x_history, iteration_count