"""
    The Newton's method to find the best fitting line. 

"""
from MethodBase import MethodBase
import numpy as np

class Newton(MethodBase):
    def __init__(self, n, points):
        super().__init__(n, points)
        self.name = "Newton's Method"

    def solve_w(self, A, b):
        """
            w1 = w0 - Hfinv(w0) * gf(w0)

        """
        w = np.zeros((self.n, 1))   # initialize w as [0, 0, ..., 0]^T
        Hf = 2 * self.multi_matrix(A.T, A)
        Hfinv = self.inv_matrix(Hf)
        gfx = 2 * self.multi_matrix(self.multi_matrix(A.T, A), w) - 2 * self.multi_matrix(A.T, b)
        w = w - self.multi_matrix(Hfinv, gfx)
        return w

    def run(self):
        A, b = self.gen_A_and_b()
        return self.solve_w(A, b)

    
