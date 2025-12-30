import multiprocessing
from itertools import product

class MyClass:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    @staticmethod
    def _worker(x, y, multiplier):
        return (x * y) * multiplier

    def run(self, xs, ys):
        # Prepare args including the constant multiplier
        args = [(x, y, self.multiplier) for x, y in product(xs, ys)]
        
        with multiprocessing.Pool() as pool:
            return pool.starmap(self._worker, args)

myc = MyClass(3)
print(myc.run([1, 2], [3, 4]))
