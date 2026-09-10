import numpy as np
import logging  


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def approximate_value(target, approximator):
    if abs(target) < 1e-6:
        return 0.0
    ladder = approximator.solve(target=target)
    # if ladder is None:
    #     logger.warning(f"No ladder found for target={target:.6f}")
    #     return target
    if approximator.N > 1:
        encoding = approximator.encode_target(target=target, indices=ladder["indices"])
        approx = encoding["total"]
        error = encoding["error"]
        return approx, error
    else:
        noisy_approx, error = approximator.add_noise(ladder, target)
        return noisy_approx, error



def approximate_model(flat_array, approximator):
    approximated = np.empty(flat_array.shape)
    errors = np.empty(flat_array.shape)
    total = len(flat_array)
    logger.info(f"Starting approximation of {total} weights")
    for i, value in enumerate(flat_array):
        approx, error = approximate_value(value, approximator)
        approximated[i] = approx
        errors[i] = error
        if i % 20 == 0: logger.info(f"Processed {i}/{total}")
    return approximated, errors
