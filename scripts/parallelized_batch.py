import subprocess
import sys
import logging
from concurrent.futures import ProcessPoolExecutor
 
N = 100

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def run_seed(seed):

    outputs = []

    cmd = [sys.executable, "-u", "reproduction_script.py", "--seed", str(seed)]

    logger.info(f"Executing reproduction_script.py with seed {seed}")

    subprocess.run(cmd, capture_output=True, text=True, check=True)

    logger.info("Executing script.py")    
        
    cmd = [sys.executable, "-u", "script.py"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    out = result.stdout.strip()
    outputs.append(out)

    logger.info(f"Finished procedure with seed {seed}")    

    return f"{seed},{'|'.join(outputs)}"

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=4) as ex:

        with open("risultati.txt", "w") as f:
            
            for row in ex.map(run_seed, range(N)):
                f.write(row + '\n')