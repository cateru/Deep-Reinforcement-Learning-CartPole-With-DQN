import logging
import os
import sys
import numpy as np
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from approximation.convert_pth import pth_to_numpy, numpy_to_pth
from approximation.workflow import approximate_model
from approximation.approximation import TargetSolver


# parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, required=True)
# args = parser.parse_args()

hyperparams = {
    "N_crosspoints": 1,
    "std_noise": 0.05,
    "seed": 42,#args.seed,
}



logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

        logger.info("Starting conversion of PyTorch model to NumPy array")

        flat_array, metadata = pth_to_numpy("../../../best_model_seed_50.pth")

        logger.info("Completed conversion to NumPy array")
        logger.info("Starting approximation of model weights")

        approximator = TargetSolver(hyperparams["N_crosspoints"], std_noise=hyperparams["std_noise"], seed=hyperparams["seed"])
        
        found_approximation, errors = approximate_model(flat_array, approximator = approximator)

        logger.info("Completed approximation of model weights")
        logger.info(f"Total mean relative error of approximation: {np.mean(errors):.6f} %")
        logger.info(f"Total standard deviation of relative error: {np.std(errors):.6f} %")

        subset_approx = []
        subset_error = []

        for i in range(len(errors)):
            if abs(found_approximation[i]) < 1:
                subset_approx.append(found_approximation[i])
                subset_error.append(errors[i])

        logger.info(f"Subset mean relative error of approximation: {np.mean(subset_error):.6f} %")
        logger.info(f"Subset standard deviation of relative error: {np.std(subset_error):.6f} %")

        logger.info("Starting conversion of approximated model to PyTorch format")

        numpy_to_pth(found_approximation, metadata, reference_model_path="../../../best_model_seed_50.pth", output_path="../single_model_riserva.pth")

        logger.info("Completed conversion of approximated model to PyTorch format")
        logger.info("All steps completed successfully")