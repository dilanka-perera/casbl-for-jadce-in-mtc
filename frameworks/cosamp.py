import numpy as np

def cosamp(
    theta: np.ndarray,
    y: np.ndarray,
    sparsity_level: int,
    max_iter: int = 500,
    stopping_criterion: float = 1e-4,
    lambda_reg: float = 1e-2
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Compressive Sampling Matching Pursuit (CoSaMP) for Multiple Measurement Vectors (MMV).

    Parameters
    ----------
    theta : np.ndarray
        Pilot matrix (L x N).
    y : np.ndarray
        Received signal matrix (L x M).
    sparsity_level : int
        Target joint sparsity level K (number of active rows in the solution).
    max_iter : int, optional
        Maximum number of iterations (default=500).
    stopping_criterion : float, optional
        Convergence tolerance (default=1e-4).
    lambda_reg : float, optional
        Tikhonov (ridge) regularization added to least-squares normal equations
        for numerical stability (default=1e-2).

    Returns
    -------
    x_hat : np.ndarray
        Final estimated signal matrix.
    x_hat_history : list[np.ndarray]
        History of estimates per iteration.
    iteration_count : int
        Number of iterations until convergence.
    """

    M = y.shape[1]
    N = theta.shape[1]

    x_hat = np.zeros((N, M), dtype=np.complex128)
    residual = y.copy()
    support = set()

    x_hat_history = []
    iteration_count = max_iter

    for t in range(max_iter):
        # Compute proxy
        proxy = theta.conj().T @ residual
        row_energy = np.sum(np.abs(proxy) ** 2, axis=1)
        omega = np.argpartition(row_energy, -2 * sparsity_level)[-2 * sparsity_level:]

        # Merge with previous support
        merged_support = list(set(omega).union(support))
        theta_s = theta[:, merged_support]

        # Regularized LS on merged support
        A = theta_s
        AHA = A.conj().T @ A + lambda_reg * np.eye(A.shape[1])
        AHY = A.conj().T @ y
        x_temp = np.linalg.solve(AHA, AHY)

        # Prune to top-K rows
        row_energy_temp = np.sum(np.abs(x_temp) ** 2, axis=1)
        top_indices = np.argpartition(row_energy_temp, -sparsity_level)[-sparsity_level:]
        support = set([merged_support[i] for i in top_indices])

        # Final LS on pruned support
        theta_final = theta[:, list(support)]
        A = theta_final
        AHA = A.conj().T @ A + lambda_reg * np.eye(A.shape[1])
        AHY = A.conj().T @ y
        x_support = np.linalg.solve(AHA, AHY)

        # Construct full x_hat
        x_hat.fill(0)
        for i, idx in enumerate(support):
            x_hat[idx, :] = x_support[i, :]

        # Update residual
        new_residual = y - theta @ x_hat

        # Save x_hat history
        x_hat_history.append(x_hat.copy())

        # Check stopping
        if np.linalg.norm(new_residual) < stopping_criterion:
            iteration_count = t + 1
            print(f"CoSaMP converged at iteration {iteration_count}")
            break

        residual = new_residual

    # Pad histories if convergence occurred early
    pad_len = max_iter - len(x_hat_history)
    if pad_len > 0:
        x_hat_pad = [x_hat_history[-1].copy()] * pad_len
        x_hat_history.extend(x_hat_pad)

    return x_hat, x_hat_history, iteration_count