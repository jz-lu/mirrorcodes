import multiprocessing
from itertools import product

def f(x, y):
    # Example heavy calculation
    return x * y

if __name__ == "__main__":
    xs = [1, 2, 3]
    ys = [10, 20, 30]
    
    # 1. Generate all combinations of (x, y)
    # This creates: [(1, 10), (1, 20), (1, 30), (2, 10), ...]
    args = list(product(xs, ys))

    # 2. Use a Pool to distribute work across CPU cores
    # 'starmap' is specifically for functions with multiple arguments
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(f, args)

    print(results)