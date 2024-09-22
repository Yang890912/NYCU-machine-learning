"""
Use Newton's method to find the best fitting line. 

"""
from MethodBase import MethodBase
import numpy as np

class Newton(MethodBase):
    def __init__(self, n, points):
        super().__init__(n, points)
        self.name = "Newton's Method"

    def solve_w(self, A, b):
        """
        w_1 = w_0 - Hfinv(w_0) * gf(w_0)

        """
        w = np.zeros((self.n, 1))   # initialize w as [0, 0, ..., 0]^T
        Hf = 2 * self.matrix_product(A.T, A)
        Hfinv = self.matrix_inverse(Hf)
        gfx = 2 * self.matrix_product(self.matrix_product(A.T, A), w) - 2 * self.matrix_product(A.T, b)
        w = w - self.matrix_product(Hfinv, gfx)
        return w

    def run(self):
        A, b = self.gen_A_and_b()
        return self.solve_w(A, b)

    
