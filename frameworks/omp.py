import numpy as np

def omp(theta, y, sparsity_level, stopping_criterion=1e-4):
    """
    Orthogonal Matching Pursuit (OMP) for MMV.

    Returns:
    - x_hat: Final estimated signal matrix
    - x_hat_history: List of x_hat matrices (length = sparsity_level)
    """

    L, N = theta.shape
    _, M = y.shape

    residual = y.copy()
    support = []
    x_hat_history = []

    for _ in range(sparsity_level):
        # Step 1: Compute proxy
        proxy = theta.conj().T @ residual  # N x M
        row_energy = np.sum(np.abs(proxy)**2, axis=1)

        # Step 2: Exclude already selected atoms
        row_energy[support] = 0
        idx = np.argmax(row_energy)
        support.append(idx)

        # Step 3: Solve least squares on current support
        theta_s = theta[:, support]  # L x |S|
        x_temp, _, _, _ = np.linalg.lstsq(theta_s, y, rcond=None)  # |S| x M

        # Step 4: Update residual
        residual = y - theta_s @ x_temp

        # Step 5: Construct full x_hat
        x_hat = np.zeros((N, M), dtype=np.complex128)
        for i, s_idx in enumerate(support):
            x_hat[s_idx, :] = x_temp[i, :]

        # Step 6: Save history
        x_hat_history.append(x_hat.copy())

    return x_hat, x_hat_history