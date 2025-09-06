import numpy as np

def omp(
    theta: np.ndarray,
    y: np.ndarray,
    sparsity_level: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Orthogonal Matching Pursuit (OMP) for Multiple Measurement Vectors (MMV).

    Parameters
    ----------
    theta : np.ndarray
        Pilot matrix (L x N).
    y : np.ndarray
        Received signal matrix (L x M).
    sparsity_level : int
        Target joint sparsity level K (number of active rows in the solution).
    
    Returns
    -------
    x_hat : np.ndarray
        Final estimated signal matrix.
    x_hat_history : list[np.ndarray]
        History of estimates per iteration.
    """

    M = y.shape[1]
    N = theta.shape[1]

    residual = y.copy()
    support = []
    x_hat_history = []

    for _ in range(sparsity_level):
        # Compute proxy
        proxy = theta.conj().T @ residual  # N x M
        row_energy = np.sum(np.abs(proxy)**2, axis=1)

        # Exclude already selected atoms
        row_energy[support] = 0
        idx = np.argmax(row_energy)
        support.append(idx)

        # Solve least squares on current support
        theta_s = theta[:, support]  # L x |S|
        x_temp, _, _, _ = np.linalg.lstsq(theta_s, y, rcond=None)  # |S| x M

        # Update residual
        residual = y - theta_s @ x_temp

        # Construct full x_hat
        x_hat = np.zeros((N, M), dtype=np.complex128)
        for i, s_idx in enumerate(support):
            x_hat[s_idx, :] = x_temp[i, :]

        # Save x_hat history
        x_hat_history.append(x_hat.copy())

    return x_hat, x_hat_history