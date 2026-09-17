from random import seed
import subprocess
import sys
import logging
from concurrent.futures import ProcessPoolExecutor
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(REPO_DIR)
from analysis.analysis import plot_rewards_from_file 


N = 100

std_d2d = [0, 0.01, 0.02, 0.05]
std_noise = [0, 0.01, 0.02, 0.05]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def run_seed(params):

    seed, std_noise, std_d2d = params

    outputs = []

    model_path = os.path.join(REPO_DIR, f"tmp_model_seed{seed}_noise{std_noise}_d2d{std_d2d}.pth")
    reproduction_script = os.path.join(SCRIPT_DIR, "reproduction_script.py")
    script = os.path.join(SCRIPT_DIR, "script.py")

    cmd = [sys.executable, "-u", reproduction_script, "--seed", str(seed), "--std_noise", str(std_noise), "--std_d2d", str(std_d2d)]

    logger.info(f"Executing reproduction_script.py with seed {seed}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info("Executing script.py")    
            
        cmd = [sys.executable, "-u", script, "--model_path", model_path]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        out = result.stdout.strip()
        outputs.append(out)

    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

    logger.info(f"Finished procedure with seed {seed}")    

    return f"{seed},{'|'.join(outputs)}"

if __name__ == "__main__":

    for noise in std_noise:
        for d2d in std_d2d:

            params_list = [(seed, noise, d2d) for seed in range(N)]

            filename = f"risultati_d2d_{d2d}_noise_{noise}.txt"

            with ProcessPoolExecutor(max_workers=1) as ex: # ProcessPoolExecutor(max_workers=4)

                with open( filename, "w") as f:
                    for row in ex.map(run_seed, params_list):
                        f.write(row + '\n')

            print(f"All simulations completed for configuration (noise={noise}, d2d={d2d}).")

            plot_rewards_from_file(filename)