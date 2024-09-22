"""
The Base of methods

"""
import numpy as np

class MethodBase:
    def __init__(self, n, points):
        self.name = "Base"
        self.n = n
        self.points = points

    def matrix_product(self, A, B):
        rlen, clen = A.shape[0], B.shape[1]
        AB = np.zeros((rlen, clen))
        for i in range(rlen):
            for j in range(clen):
                for k in range(A.shape[1]):
                    AB[i][j] += A[i][k] * B[k][j]
        return AB

    def matrix_inverse(self, M):
        """
        Find the inverse matrix by Gauss-Jordan elimination
        
        """
        rlen = M.shape[0]
        Minv = np.eye(rlen)
        for i in range(rlen):
            for m in range(rlen):
                if m != i:
                    R = M[m][i] / M[i][i]   # ratio
                    for n in range(rlen):
                        Minv[m][n] = Minv[m][n] - R * Minv[i][n]
                        M[m][n] = M[m][n] - R * M[i][n]
        
        for i in range(rlen):
            R = M[i][i]
            for n in range(rlen):
                Minv[i][n] = Minv[i][n] / R
                M[i][n] = M[i][n] / R 
        return Minv
    
    def gen_A_and_b(self):
        """
        Generate matrices A and b

        """
        row, column = len(self.points), self.n
        A, b = np.zeros((row, column)), np.zeros((row, 1))
        for i, xy in enumerate(self.points):
            x, y = xy[0], xy[1]
            for j in range(self.n-1, -1, -1):
                A[i][j] = x**(self.n-1-j)
            b[i][0] = y
        return A, b

