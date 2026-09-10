import subprocess
import sys

N = 100
scripts = [
    "reproduction_script.py",
    "script.py",
]

with open("risultati.txt", "w") as f:

    for i in range(N):

        outputs = []

        seed = i

        print(f"Iterazione {i+1}/{N}")

        for script in scripts:

            cmd = [sys.executable, "-u", script]

            if script == "reproduction_script.py":
                cmd += ["--seed", str(seed)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            out = result.stdout.strip()

            outputs.append(out)

        f.write(f"{i},{seed},{' | '.join(outputs)}\n")
        f.flush()