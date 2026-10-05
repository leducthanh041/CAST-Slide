import numpy as np


def forgetting(results: list[list[float]]) -> float:
    """
    Compute average forgetting after the final task.

    results is a triangular list of lists:
        results[0] = [bacc_t0, bacc_t1]
        results[1] = [bacc_t0, bacc_t1, bacc_t2]
        ...
        results[T-2] = [bacc_t0, ..., bacc_tT-1]
    """
    n_rows  = len(results)
    max_len = max(len(r) for r in results)

    # Pad rows before converting to a dense array.
    for i in range(n_rows):
        results[i] += [0.0] * (max_len - len(results[i]))

    np_res        = np.array(results)
    best_per_task = np.max(np_res, axis=0)

    fgt = [best_per_task[i] - results[-1][i] for i in range(n_rows)]
    return float(np.mean(fgt))


def backward_transfer(results: list[list[float]]) -> float:
    """
    Compute average Backward Transfer (BWT).

    results[i][i] is the score for task i when task i is first evaluated in the
    triangular matrix; results[-1][i] is the score after all tasks.
    """
    n_rows = len(results)   # = NUM_TASKS - 1
    bwt = [results[-1][i] - results[i][i] for i in range(n_rows)]
    return float(np.mean(bwt))


def pad_numpy_arrays(arrays: list[np.ndarray],
                     pad_value: float = 0.0) -> np.ndarray:
    """
    Pad arrays with different shapes to a common shape before stacking.
    """
    max_dim   = max(a.ndim for a in arrays)
    arrays    = [a.reshape((1,) * (max_dim - a.ndim) + a.shape) for a in arrays]
    max_shape = np.max([a.shape for a in arrays], axis=0)
    padded    = []
    for a in arrays:
        pw = [(0, max_shape[i] - a.shape[i]) for i in range(max_dim)]
        padded.append(np.pad(a, pw, constant_values=pad_value))
    return np.stack(padded)
