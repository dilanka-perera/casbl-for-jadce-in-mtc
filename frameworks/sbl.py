import numpy as np

def sbl(
    theta: np.ndarray,
    y: np.ndarray,
    noise_var: float,
    max_iter: int = 500,
    stopping_criterion: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], int]:
    """
    Sparse Bayesian Learning (SBL) for Multiple Measurement Vectors (MMV).

    Parameters
    ----------
    theta : np.ndarray
        Pilot matrix (L x N).
    y : np.ndarray
        Received signal matrix (L x M).
    noise_var : float
        Noise variance.
    max_iter : int, optional
        Maximum number of EM iterations (default=500).
    stopping_criterion : float, optional
        Convergence tolerance (default=1e-4).

    Returns
    -------
    gamma_new : np.ndarray
        Final activity vector.
    mu_z : np.ndarray
        Final posterior mean estimates of signals.
    gamma_history : list[np.ndarray]
        History of activity estimates per iteration.
    mu_z_history : list[np.ndarray]
        History of posterior mean estimates per iteration.
    iteration_count : int
        Number of iterations until convergence.
    """

    M = y.shape[1]
    L = theta.shape[0]
    N = theta.shape[1]

    # Initialize Gamma
    Gamma = np.eye(N) * 0.1

    # Initialize histories
    gamma_history: list[np.ndarray] = []
    mu_z_history: list[np.ndarray] = []
    iteration_count = max_iter  # default unless convergence occurs earlier

    for t in range(max_iter):
        # E-step
        Sigma_y = theta @ Gamma @ theta.conj().T + noise_var * np.eye(L)
        Sigma_y_inv = np.linalg.inv(Sigma_y)
        Sigma_z = Gamma - Gamma @ theta.conj().T @ Sigma_y_inv @ theta @ Gamma
        mu_z = Gamma @ theta.conj().T @ Sigma_y_inv @ y

        # Gamma update
        gamma_new = (np.linalg.norm(mu_z, axis=1) ** 2) / M + np.real(np.diag(Sigma_z))
        gamma_new = np.maximum(gamma_new, 1e-8)  # Ensure positivity / numerical stability

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
