"""
Use LSE method to find the best fitting line. 

"""
from MethodBase import MethodBase
import numpy as np

class LSE(MethodBase):
    def __init__(self, n, l, points):
        super().__init__(n, points)
        self.name = "LSE"
        self.l = l
 
    def solve_w(self, A, b):
        """
        ((A^T) * A + lambda)^(-1) * (A^T) * b

        """
        w = self.matrix_product(A.T, A) + self.l * np.eye(self.n)
        w = self.matrix_inverse(w)
        w = self.matrix_product(w, A.T)
        w = self.matrix_product(w, b)
        return w
        
    def run(self):
        A, b = self.gen_A_and_b()
        return self.solve_w(A, b)
        


    



        
