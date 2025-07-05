import numpy as np

def cosamp(theta, y, sparsity_level, max_iter=500, stopping_criterion=1e-4, lambda_reg=1e-2):
    """
    Compressive Sampling Matching Pursuit (CoSaMP) for MMV.

    Returns:
    - x_hat: Final estimated signal matrix
    - x_hat_history: List of x_hat matrices per iteration (length = max_iter)
    - iteration_count: Number of iterations executed before convergence
    """

    L, N = theta.shape
    _, M = y.shape

    x_hat = np.zeros((N, M), dtype=np.complex128)
    residual = y.copy()
    support = set()

    x_hat_history = []
    iteration_count = max_iter

    for t in range(max_iter):
        # Step 1: Compute proxy
        proxy = theta.conj().T @ residual
        row_energy = np.sum(np.abs(proxy) ** 2, axis=1)
        omega = np.argpartition(row_energy, -2 * sparsity_level)[-2 * sparsity_level:]

        # Step 2: Merge with previous support
        merged_support = list(set(omega).union(support))
        theta_s = theta[:, merged_support]

        # Step 3: Regularized LS on merged support
        A = theta_s
        AHA = A.conj().T @ A + lambda_reg * np.eye(A.shape[1])
        AHY = A.conj().T @ y
        x_temp = np.linalg.solve(AHA, AHY)

        # Step 4: Prune to top-K rows
        row_energy_temp = np.sum(np.abs(x_temp) ** 2, axis=1)
        top_indices = np.argpartition(row_energy_temp, -sparsity_level)[-sparsity_level:]
        support = set([merged_support[i] for i in top_indices])

        # Step 5: Final LS on pruned support
        theta_final = theta[:, list(support)]
        A = theta_final
        AHA = A.conj().T @ A + lambda_reg * np.eye(A.shape[1])
        AHY = A.conj().T @ y
        x_support = np.linalg.solve(AHA, AHY)

        # Step 6: Construct full x_hat
        x_hat.fill(0)
        for i, idx in enumerate(support):
            x_hat[idx, :] = x_support[i, :]

        # Step 7: Update residual
        new_residual = y - theta @ x_hat

        # Step 8: Save x_hat history
        x_hat_history.append(x_hat.copy())

        # Step 9: Check stopping
        if np.linalg.norm(new_residual) < stopping_criterion:
            iteration_count = t + 1
            print(f"CoSaMP converged at iteration {iteration_count}")
            break

        residual = new_residual

    # Step 10: Pad history if needed
    pad_len = max_iter - len(x_hat_history)
    if pad_len > 0:
        x_hat_pad = [x_hat_history[-1].copy()] * pad_len
        x_hat_history.extend(x_hat_pad)

    return x_hat, x_hat_history, iteration_count
