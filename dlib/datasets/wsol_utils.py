import numpy as np
from dlib.configure import constants

def compute_weights(n, alpha, mode, sl_tc_knn_mode=constants.TIME_BEFORE_AFTER):
    """
    Compute normalized weights for a given number of frames, alpha, and mode.

    Parameters:
    - n: Total number of frames.
    - alpha: Decay rate parameter.
    - mode: Weighting mode.
    - sl_tc_knn_mode: Specifies the weight arrangement. It can be TIME_BEFORE, TIME_AFTER, or TIME_BEFORE_AFTER.

    Returns:
    - A list of normalized weights.
    """
    c = n // 2
    
    if sl_tc_knn_mode == constants.TIME_BEFORE:
        index_func = lambda i: n - i - 1
    elif sl_tc_knn_mode == constants.TIME_AFTER:
        index_func = lambda i: i
    elif sl_tc_knn_mode == constants.TIME_BEFORE_AFTER:
        index_func = lambda i: abs(c - i)
    else:
        raise ValueError(f"Invalid sl_tc_knn_mode value: {sl_tc_knn_mode}")

    if mode == 'exp':
        weights = [np.exp(-alpha * index_func(i)) for i in range(n)]
    elif mode == 'log':
        weights = [1 / (1 + alpha * index_func(i)) for i in range(n)]
    elif mode == 'linear':
        weights = [1 - alpha * index_func(i) / n for i in range(n)]
    elif mode == 'quadratic':
        weights = [1 - alpha * index_func(i)**2 / n**2 for i in range(n)]
    elif mode == 'inverse_square':
        weights = [1 / (1 + alpha * index_func(i)**2 / n**2) for i in range(n)]
    elif mode == 'sigmoid':
        offset = 1 / (1 + np.exp(alpha * c))
        weights = [(1 / (1 + np.exp(alpha * (index_func(i) - c))) - offset) for i in range(n)]
    elif mode == 'gaussian':
        weights = [np.exp(-alpha * index_func(i)**2 / n**2) for i in range(n)]
    elif mode == 'cosine':
        weights = [(1 + np.cos(np.pi * index_func(i) / n)) / 2 for i in range(n)]
    
    return np.array(weights) / sum(weights)

# def compute_weights_for_weighted_avg_of_cam(sl_tc_knn, alpha, weight_distribution, sl_tc_knn_mode):
#     """
#     Compute normalized weights for a given number of frames, alpha, and weight distribution.

#     Parameters:
#     - sl_tc_knn: Total number of frames.
#     - alpha: Decay rate parameter.
#     - weight_distribution: Type of weight distribution. 
#     - sl_tc_knn_mode: Specifies the weight arrangement. It can be TIME_BEFORE, TIME_AFTER, or TIME_BEFORE_AFTER.

#     Returns:
#     - A list of normalized weights.
#     """
    
#     if sl_tc_knn_mode == constants.TIME_BEFORE_AFTER:
#         total_frames = 2 * sl_tc_knn - 1
#     else:
#         total_frames = sl_tc_knn
    
#     c = total_frames // 2
    
#     if sl_tc_knn_mode == constants.TIME_BEFORE:
#         index_func = lambda i: sl_tc_knn - i - 1
#     elif sl_tc_knn_mode == constants.TIME_AFTER:
#         index_func = lambda i: i
#     elif sl_tc_knn_mode == constants.TIME_BEFORE_AFTER:
#         index_func = lambda i: abs(c - i)
#     else:
#         raise ValueError(f"Invalid sl_tc_knn_mode value: {sl_tc_knn_mode}")

#     if weight_distribution == 'exp':
#         weights = [np.exp(-alpha * index_func(i)) for i in range(total_frames)]
#     elif weight_distribution == 'log':
#         weights = [1 / (1 + alpha * index_func(i)) for i in range(total_frames)]
#     elif weight_distribution == 'linear':
#         weights = [1 - alpha * index_func(i) / total_frames for i in range(total_frames)]
#     elif weight_distribution == 'quadratic':
#         weights = [1 - alpha * index_func(i)**2 / total_frames**2 for i in range(total_frames)]
#     elif weight_distribution == 'inverse_square':
#         weights = [1 / (1 + alpha * index_func(i)**2 / total_frames**2) for i in range(total_frames)]
#     elif weight_distribution == 'sigmoid':
#         offset = 1 / (1 + np.exp(alpha * c))
#         weights = [(1 / (1 + np.exp(alpha * (index_func(i) - c))) - offset) for i in range(total_frames)]
#     elif weight_distribution == 'gaussian':
#         weights = [np.exp(-alpha * index_func(i)**2 / total_frames**2) for i in range(total_frames)]
#     elif weight_distribution == 'cosine':
#         weights = [(1 + np.cos(np.pi * index_func(i) / total_frames)) / 2 for i in range(total_frames)]
#     else:
#         raise ValueError(f"Invalid weight_distribution value: {weight_distribution}")
    
#     return np.array(weights) / sum(weights)