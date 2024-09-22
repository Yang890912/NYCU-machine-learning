"""
Use steepest descent method to find the best fitting line. 

"""
from MethodBase import MethodBase
import numpy as np

class SteepestDescent(MethodBase):
    def __init__(self, n, lr, points):
        super().__init__(n, points)
        self.name = "Steepest Descent Method"
        self.lr = lr
 
    def solve_w(self, A, b):
        """
        w_1 = w_0 - lr * gf(w_0)

        """
        w = np.zeros((self.n, 1))   # initialize w as [0, 0, ..., 0]^T
        epochs = 10000
        for _ in range(epochs):
            gfx = 2 * self.matrix_product(self.matrix_product(A.T, A), w) - 2 * self.matrix_product(A.T, b)
            w = w - self.lr * gfx
        return w

    def run(self):
        A, b = self.gen_A_and_b()
        return self.solve_w(A, b)
