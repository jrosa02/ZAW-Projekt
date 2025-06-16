import trajEstimaton
from multiprocessing import Pool

import time
from contextlib import contextmanager

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
        fy=parameters[1],
        cx=parameters[2],
        cy=parameters[3],
        k1=parameters[4],
        k2=parameters[5],
        p1=parameters[6],
        p2=parameters[7],
        k3=parameters[8],
        filter_data=parameters[9],
        filter_t=parameters[10],
        filter_k=parameters[11],
        filter_size=parameters[12]
    )
    estimator.process_event_frames(tau=0.25)

    return hash(estimator)

class fastOptimization():
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def run_parallel(self, parameter_set, num_workers=4):
        args = [(path, parameter_set) for path in self.filepaths]  # Repeat same parameter_set
        with Pool(num_workers) as pool:
            results = pool.map(single_file_cost, args)
        return results


if __name__ == "__main__":
    for n in range(1, 5):
        print("trying ", n)
        fast_opt = fastOptimization([f"data/train/{i:04}.npz" for i in range(n)])
        with timer("Multithread"):
            for _ in range(4):
                print(fast_opt.run_parallel(parameter_set=[217.2, 277, 100, 100, 0, 0, 0, 0, 0, False, 1/24, 1, 5]))
    

