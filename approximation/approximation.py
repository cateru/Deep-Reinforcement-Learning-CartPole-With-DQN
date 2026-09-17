import bisect
import logging
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TargetSolver:
    def __init__(self, N, std_noise, std_d2d=0.05, seed=34):
        self.rng = np.random.default_rng(seed)
        self.pulses = np.arange(1, 15001)
        self.N = N        
        self.d2d_a = self.rng.normal(loc = 1, scale=std_d2d, size=N)
        self.d2d_b = self.rng.normal(loc = 1, scale=std_d2d, size=N)
        self.d2d_c = self.rng.normal(loc = 1, scale=std_d2d, size=N)
        self.random_noise = self.rng.normal(loc = 1, scale=std_noise, size=(N, len(self.pulses) + 1))
        self.GP = np.array([[self.conductance_P(p, device=i) for p in self.pulses] for i in range(self.N)])
        self.GAP = np.array([[self.conductance_AP(p, device=i) for p in self.pulses] for i in range(self.N)])
        self.Gin = np.array([[self.intermediate_conductance(p, device=i) for p in self.pulses] for i in range(self.N)])
        self.dG = self.GAP - self.GP
        self.GAP = np.array([[self.conductance_AP(p, same_noise=False, device=i) for p in self.pulses] for i in range(self.N)])
        self.translate_conductance()
        # self.build_pair_sums()
        self.baseline = np.sum(self.Gin[:, 0]*self.d2d_c)

    def log_func(self, pulse_number, a = 2, b = 0):
        return a * np.log10(pulse_number) + b

    def conductance_P(self, pulse_number, device = 0, linear = False):
        stretch = self.d2d_a[device]
        shift = self.d2d_b[device]
        if linear:            
            line =  0.001 * pulse_number
            return line * stretch * self.random_noise[device, pulse_number] + shift * 0.001
        else:
            log_func = self.log_func(pulse_number)
            return log_func * stretch * self.random_noise[device,pulse_number] + shift * self.log_func(self.pulses[0])

    def conductance_AP(self, pulse_number, same_noise = True, device = 0, linear = False):
        stretch = self.d2d_a[device]
        shift = self.d2d_b[device]
        if linear: 
            Gap = 0.0012 * pulse_number
            return Gap * stretch * self.random_noise[device,pulse_number] + shift * 0.0012
        else:
            c = 3
            x = pulse_number - c
            Gap = self.log_func(x, a = 2.2)
            if same_noise:
                return (Gap + (self.random_noise[device,pulse_number]-1) * self.log_func(pulse_number)) * stretch + shift * self.log_func(self.pulses[0])
            else:
                return Gap * stretch * self.random_noise[device,pulse_number] + shift * self.log_func(self.pulses[0], a = 2.2)

    def intermediate_conductance(self, pulse_number, device = 0):
        log = self.log_func(pulse_number, a = 2.1)
        Gin = log * self.d2d_a[device] + self.d2d_b[device] * self.log_func(self.pulses[0], a = 2.1)
        # stretch = self.d2d_a[device]
        # shift = self.d2d_b[device]
        # Gin = 0.001 * pulse_number * stretch * self.random_noise[device, pulse_number] + shift * 0.001
        return Gin 

    def translate_conductance(self):
        start_index = 14
        self.pulses = self.pulses[start_index:]
        self.GP = self.GP[:, start_index:]
        self.GAP = self.GAP[:, start_index:]
        self.dG = self.dG[:, start_index:]
        self.Gin = self.Gin[:, start_index:]

    def closest_index(self, value, device=0):
        arr = self.dG[device]
        pos = bisect.bisect_left(arr, value)
        if pos == 0:
            return 0
        if pos >= len(arr):
            return len(arr) - 1
        before = pos - 1
        after = pos
        if abs(arr[after] - value) < abs(arr[before] - value):
            return after
        return before

    def build_from_base(self, base_idx):
        base_delta = self.dG[0, base_idx]
        indices = []
        total = 0.0
        max_prop_error = 0.0
        for device in range(self.N):
            desired = base_delta * (2 ** device)
            k = self.closest_index(desired, device)
            actual = self.dG[device, k]
            prop_error = abs(actual - desired) / abs(desired)
            max_prop_error = max(
                max_prop_error,
                prop_error
            )
            indices.append((device, k))
            total += self.Gin[device, k]   #before GP
        return {
            "indices": indices,
            "total": total,
            "base_delta": base_delta,
            "max_prop_error (%)": max_prop_error
        }

    
    def encode_target(self, target, indices):
        sign = np.sign(target)
        G_pairs = [(self.GP[device, k], self.GAP[device, k]) for device, k in indices]
        best = None
        best_error = float("inf") 
        for cfg in product(*G_pairs): 
            goal = sum(cfg) 
            if sign > 0:
                approx = goal - self.baseline
            else:
                approx = -goal + self.baseline
            rel_error = (abs(approx - target) / abs(target))*100
            if rel_error < best_error: 
                best_error = rel_error 
                best = {
                    "values": cfg,
                    "total": approx,
                    "error": rel_error
                }
                logger.info(f"Final combination value: {approx} | And error (%): {rel_error}")
            # logger.info(f"goal={goal:.6f} | total={total:.6f} | rel_error={rel_error:.6f}")
            return best
        
    def add_noise(self, sol, target):
        idx = sol["indices"]
        sign = np.sign(target)
        if self.N < 2:
            device, k = idx[0]
            noisy_goal = self.Gin[device, k] * self.random_noise[device,k]
        else: 
            noisy_vals = [self.GP[device, k] * self.random_noise[device,k] for device, k in idx]
            noisy_goal = sum(noisy_vals)
        if sign > 0:
            noisy_total = noisy_goal - self.baseline
        else:
            noisy_total = -noisy_goal + self.baseline
        rel_error = (abs(noisy_total - target) / abs(target))*100
        return noisy_total, rel_error


    def solve(self, target):
        sign = np.sign(target)
        if sign < 0:
            goal = self.baseline - target
        else: 
            goal =  self.baseline + target
        logger.info(f"Starting solve | target={target:.4f} | goal={goal:.4f} | N={self.N}")
        best = None
        best_error = float("inf")
        prev_error = float("inf")
        for base_idx in range(len(self.pulses)):
            sol = self.build_from_base(base_idx)
            # logger.info(f"indexes={sol["indices"]} |" f"goal={sol["total"]}")
            total = sol["total"]
            if sign > 0:
                total = total - self.baseline
            else:
                total = -total + self.baseline
            rel_error = (abs(total - target) / abs(target))
            # logger.info(f"total={total:.6f} | " f"rel_error={rel_error:.6f}")
            if rel_error > prev_error:
                logger.info("Increasing error, stopping search")
                break
            prev_error = rel_error
            if rel_error < best_error:
                best_error = rel_error
                best = sol.copy()    
                if sign > 0:
                    best["total"] = best["total"] - self.baseline
                else:
                    best["total"] = -best["total"] + self.baseline         
                # logger.info(f"NEW BEST | "f"base_idx={base_idx} | "f"error={best_error:.6f}")
        if best is not None:
            best["target_relative_error"] = best_error
            # logger.info(f"Solution found | "f"error={best_error:.6f}")
        else: logger.warning("No valid solution found")
        logger.info(f"Best solution | target={best['total']:.6f} | error (%)={best['target_relative_error']:.6f} | indices={best['indices']}")
        return best
    
                
    def build_pair_sums(self):
        A = np.asarray(self.GP)
        n = len(A)
        i_idx = np.repeat(np.arange(n), n)
        j_idx = np.tile(np.arange(n), n)
        sums = A[i_idx] + A[j_idx]
        order = np.argsort(sums)
        self.pair_values = sums[order]
        self.pair_i = i_idx[order]
        self.pair_j = j_idx[order]


    def solve_no_MR(self, target):
        sign = np.sign(target)
        if sign < 0:
            goal = self.baseline - target
        else:
            goal = self.baseline + target
        A = np.asarray(self.GP)
        n = len(A)
        pair_values = self.pair_values
        pair_i = self.pair_i
        pair_j = self.pair_j
        best_error = float("inf")
        best = None
        for k in range(n):
            ak = A[k]
            needed = goal - ak
            idx = np.searchsorted(pair_values, needed)
            for cand in (idx - 1, idx, idx + 1):
                if 0 <= cand < len(pair_values):
                    total = pair_values[cand] + ak
                    i = pair_i[cand]
                    j = pair_j[cand]
                    if sign > 0:
                        total_adj = total - self.baseline
                    else:
                        total_adj = -total + self.baseline
                    rel_error = (abs(total_adj - target) / abs(target)) * 100
                    if rel_error < best_error:
                        best_error = rel_error
                        best = {
                            "indices": (i, j, k),
                            "total": total_adj,
                            "target_relative_error": best_error,
                        }
        return best
    
    def plot_arrays(self, device=0):
        _, ax = plt.subplots(1, 2, figsize=(10, 5)) 
        ax[0].plot(self.pulses, self.GP[device].T, label="GP", color="blue", linewidth=2)
        ax[0].plot(self.pulses, self.GAP[device].T, label="GAP", color="red", linewidth=2)
        ax[0].set_xlabel("Pulse Number")
        ax[0].set_ylabel("Conductance")
        ax[1].plot(self.pulses, self.dG[device].T, color="tab:green", linewidth=2)
        ax[1].set_xlabel("Pulse Number")
        ax[1].set_ylabel("dG")
        ax[0].legend()
        plt.tight_layout()
        plt.show()

    def plot_solution(self, sol):
        indices = sol["indices"]
        actual = [self.dG[device, k] for device, k in indices]
        expected = [sol["base_delta"] * (2 ** i)  for i in range(len(indices)) ] 
        print("Expected:", expected)
        print("Actual:", actual)
        plt.figure(figsize=(10,6)) 
        plt.plot( expected, 'o-', label='Expected' ) 
        plt.plot( actual, 'x--', label='Actual' ) 
        plt.yscale("log") 
        plt.xlabel("Term") 
        plt.ylabel("dG")  
        plt.legend()  
        plt.show()

    def plot_encoding(self, encoding, target=None):
        if target > 0:
            target = self.baseline + target
            total = encoding["total"] + self.baseline
        else:
            target = self.baseline - target
            total = -encoding["total"] + self.baseline
        vals = encoding["values"]
        cumulative = np.cumsum(vals) 
        colors = ["tab:blue" if v in self.GP else "tab:red"for v in vals]
        plt.figure(figsize=(12,5))
        plt.bar(range(len(vals)), vals, label="Selected conductances", color=colors, alpha=0.4, edgecolor=colors)
        plt.plot(range(len(vals)), cumulative, marker='o', label="Cumulative sum")
        plt.axhline(total, linestyle="--", label=f"Total = {total:.3f}")
        plt.axhline(target, linestyle=":", label=f"Target = {target:.3f}")
        plt.xlabel("Bit")
        plt.ylabel("Conductance")
        plt.legend()
        plt.tight_layout()
        plt.show()