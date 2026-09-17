import logging
import os
import sys
import numpy as np
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
from approximation.convert_pth import pth_to_numpy, numpy_to_pth
from approximation.workflow import approximate_model
from approximation.approximation import TargetSolver


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--std_noise", type=float, required=True)
parser.add_argument("--std_d2d", type=float, required=True)

args = parser.parse_args()

hyperparams = {
    "N_crosspoints": 3,
    "std_noise": args.std_noise,
    "std_d2d": args.std_d2d,
    "seed": args.seed,
}



logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

        logger.info("Starting conversion of PyTorch model to NumPy array")

        noise_str = f"{float(args.std_noise):g}"
        d2d_str = f"{float(args.std_d2d):g}"

        reference_model_path = os.path.join(REPO_DIR, "best_model_seed_49.pth")

        flat_array, metadata = pth_to_numpy(reference_model_path)

        logger.info("Completed conversion to NumPy array")
        logger.info("Starting approximation of model weights")

        approximator = TargetSolver(hyperparams["N_crosspoints"], std_noise=hyperparams["std_noise"], std_d2d=hyperparams["std_d2d"], seed=hyperparams["seed"])
        
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

        temp_model_path = os.path.join(REPO_DIR,f"tmp_model_seed{args.seed}_noise{noise_str}_d2d{d2d_str}.pth")

        numpy_to_pth(found_approximation, metadata, reference_model_path=reference_model_path, output_path=temp_model_path)

        logger.info("Completed conversion of approximated model to PyTorch format")
        logger.info("All steps completed successfully")