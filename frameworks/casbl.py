import numpy as np

def casbl(theta, y, noise_var, loc,
                        alpha=1.00, beta=0.1, rho=7, U=20,
                        max_iter=500, stopping_criterion=1e-4,
                        gamma_update_mode="clip", tau=0.05):
    """
    Correlation-Aware Sparse Bayesian Learning (CASBL) for MMV.

    Returns:
    - gamma_new: Final soft gamma vector (pre-threshold)
    - mu_z: Final posterior mean estimate
    - gamma_history: List of gamma vectors (length = max_iter)
    - mu_z_history: List of mu_z matrices (length = max_iter)
    - iteration_count: Number of iterations actually executed
    """

    M = y.shape[1]
    L = theta.shape[0]
    N = theta.shape[1]

    # Initialize Gamma
    Gamma = np.eye(N) * 0.1

    # Spatial correlation matrix
    if rho == 0:
        correlation_matrix = np.eye(N)
    else:
        distance_matrix = np.linalg.norm(loc[:, np.newaxis, :] - loc[np.newaxis, :, :], axis=2)
        correlation_matrix = np.maximum(
            (np.exp(-distance_matrix / rho) - np.exp(-U / rho)) / (1 - np.exp(-U / rho)), 0
        )

    # Initialize histories
    gamma_history = []
    mu_z_history = []
    iteration_count = max_iter  # default unless convergence occurs earlier

    for t in range(max_iter):
        # E-step
        Sigma_y = theta @ Gamma @ theta.conj().T + noise_var * np.eye(L)
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_z = Gamma - Gamma @ theta.conj().T @ Sigma_y_inv @ theta @ Gamma
        mu_z = Gamma @ theta.conj().T @ Sigma_y_inv @ y

        # Q and P
        Q = (np.linalg.norm(mu_z, axis=1) ** 2) / M + np.diag(Sigma_z)
        P = 2 * alpha * (beta - correlation_matrix) @ np.diag(Gamma)
        P = np.maximum(P, 1e-8)

        # Gamma update
        gamma_new = (np.sqrt(1 + 4 * P * Q) - 1) / (2 * P)
        gamma_new = np.real(gamma_new)

        # Map gamma values to [0, 1] using chosen activation strategy
        if gamma_update_mode == "clip":
            gamma_new = np.clip(gamma_new, 0, 1)
        elif gamma_update_mode == "sigmoid":
            gamma_new = 1 / (1 + np.exp(-gamma_new))
        elif gamma_update_mode == "binary":
            gamma_new = (gamma_new >= tau).astype(float)
        else:
            raise ValueError("gamma_update_mode must be 'clip', 'sigmoid', or 'binary'")

        # Save history
        mu_z_history.append(mu_z.copy())
        gamma_history.append(gamma_new.copy())

        # Convergence check
        gamma_old = np.diag(Gamma)
        if np.linalg.norm(gamma_old - gamma_new) < stopping_criterion:
            iteration_count = t + 1
            print(f"Converged after {iteration_count} iterations")
            break

        # Update Gamma
        Gamma = np.diagflat(gamma_new)

    # Pad histories if convergence occurred early
    pad_len = max_iter - len(gamma_history)
    if pad_len > 0:
        gamma_pad = [gamma_history[-1].copy()] * pad_len
        mu_z_pad = [mu_z_history[-1].copy()] * pad_len
        gamma_history.extend(gamma_pad)
        mu_z_history.extend(mu_z_pad)

    return gamma_new, mu_z, gamma_history, mu_z_history, iteration_count