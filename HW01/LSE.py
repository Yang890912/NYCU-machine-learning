"""
    The LSE method to find the best fitting line. 

"""
from MethodBase import MethodBase
import numpy as np

class LSE(MethodBase):
    def __init__(self, n, l, points):
        super().__init__(n, points)
        self.name = "LSE"
        self.l = l
 
    def solve_w(self, A, b):
        _A = self.multi_matrix(A.T, A) + self.l*np.eye(self.n)
        _A = self.inv_matrix(_A)
        w = self.multi_matrix(_A, A.T)
        w = self.multi_matrix(w, b)
        return w
        
    def run(self):
        A, b = self.gen_A_and_b()
        return self.solve_w(A, b)
        


    



        
