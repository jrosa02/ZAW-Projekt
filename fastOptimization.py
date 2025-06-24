from typing import List
import trajEstimaton
from multiprocessing import Pool

import time
from contextlib import contextmanager
import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult

@contextmanager
def timer(name="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] Execution time: {end - start:.4f} seconds")


def single_file_cost(args):
    path, parameters = args
    estimator = trajEstimaton.TrajEstimator(
        npz_path=path,
        fx=parameters[0],
        cx=100,
        cy=100,
        filter_output_pose=True,
        output_filter_cutoff=parameters[1],
        vertical_scaling_factor=parameters[2],
        filter_rangemeter=True,
    )
    cost = estimator.process_event_frames(tau=parameters[3])
    # print(f'cost: {cost}')
    return cost

class fastOptimization():
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def run_parallel(self, parameter_set):
        args = [(path, parameter_set) for path in self.filepaths]  # Repeat same parameter_set
        num_workers = len(args)
        with Pool(num_workers) as pool:
            results: List[np.float64] = pool.map(single_file_cost, args) # type: ignore
            
        results = np.sum(np.absolute(results))
        return results

    def optimize_parameters(self, initial_guess, bounds, max_iter):
        """
        Optimizes the run_parallel method using given initial guess and bounds.

        :param initial_guess: List or array of initial parameter values.
        :param bounds: Sequence of (min, max) pairs for each element in parameter_set.
        :param max_iter: Maximum number of iterations.
        :return: Optimization result containing optimal parameters and minimized value.
        """
        best = {
            "x": initial_guess,
            "fun": float("inf")
        }

        options = {
            'maxiter': max_iter,
            'disp': True,
            # 'maxls': 50,
        }

        def objective(x):
            with timer("MultiProcess"):
                cost = self.run_parallel(x)
            print(f"Cost: {cost}")
            print(f"Parameters: {x}")
            # Update best-so-far
            if cost < best["fun"]:
                best["x"] = x.copy()
                best["fun"] = cost
            return cost

        def callback(xk):
            # Evaluate and track cost without timer to avoid double timing
            cost = self.run_parallel(xk)
            if cost < best["fun"]:
                best["x"] = xk.copy()
                best["fun"] = cost

        try:
            result = optimize.minimize(
                objective,
                x0=initial_guess,
                bounds=bounds,
                method='Powell',
                callback=callback,
                options=options
            )
        except Exception or KeyboardInterrupt as e:
            print("\nOptimization interrupted. Returning best-so-far solution.")
            print(e)
            result = OptimizeResult({
                'x': best['x'],
                'fun': best['fun'],
                'message': 'Interrupted by user',
                'success': False
            })

        return result


if __name__ == "__main__":
    filepaths = [f"data/train/{i:04}.npz" for i in range(28)]
    fast_opt = fastOptimization(filepaths)

    make_bounds = lambda vals: [
    (v - abs(v) * 0.5, v + abs(v) * 0.5) if v != 0 else (-0.01, 0.01)
    for v in vals
    ]
    
    with timer("MultiProcess"):
        initial = [200, 0.07, 1.8, 0.35]
        bounds = make_bounds(initial)

        bounds[0] = (50, 500)
        bounds[1] = (0.04, 0.3)
        bounds[2] = (1, 3.8)
        bounds[3] = (0.05, 0.9)
        print(fast_opt.optimize_parameters(initial, bounds, max_iter=10))
    

