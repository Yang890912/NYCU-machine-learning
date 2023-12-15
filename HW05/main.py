import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from libsvm.svmutil import *

def load_data(filename):
    X = []
    Y = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            X.append(float(data[0]))
            Y.append(float(data[1]))
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y).reshape(-1, 1)
    return X, Y

def load_csv(filename):
    data = []
    with open(filename, 'r', newline='') as f:
        rows = csv.reader(f)
        for row in rows:
            r = list(map(float, row))
            data.append(r)
    data = np.asarray(data)
    return data

def mul(A, B):
    return np.matmul(A, B)

def inv(A):
    return np.linalg.inv(A)

class GaussianProcess:
    def __init__(self, trainX, trainY, beta=5):
        self.train_x = trainX
        self.train_y = trainY
        self.num_of_test = 1000
        self.beta = beta
        self.alpha = 1
        self.l = 1

    def cal_cov(self, x):
        """
        Calculate the coveriance

        :return: coveriance
        :rtype: np.array
        :param x: training data points
        :type x: np.array
        """
        dim = len(x)
        return self.kernel(x, x) + np.eye(dim) * 1/self.beta

    def kernel(self, x1, x2):
        """
        Rational Quadratic Kernel

        :return: the kernel
        :rtype: np.array
        :param x1: data points x
        :type x1: np.array
        :param x2: data points x
        :type x2: np.array
        """
        alpha, l = self.alpha, self.l
        dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return (1 + dist/(2*alpha*l*l)) ** (-alpha)
    
    def run(self):
        # predict with initial parameter setting (alpha = 1, l = 1)
        self.perdict()  

        # minimize negative log likelihood
        self.opt()  

        self.perdict()
   
    def perdict(self):
        test_x = np.linspace(-60, 60, self.num_of_test).reshape(-1, 1)
        C = self.cal_cov(self.train_x)

        # pre-calcualte
        K = self.kernel(self.train_x, test_x)

        # cal mu
        mu = mul(mul(K.T, inv(C)), self.train_y)

        # cal cov
        k_star = self.kernel(test_x, test_x) + 1/self.beta
        cov = k_star - mul(mul(K.T, inv(C)), K)

        def plot(mu, cov):
            plt.style.use('fast')
            fig = plt.figure(figsize=(6,4))
            axe = fig.add_subplot()
            axe.set_xlim([-60, 60])
            axe.set_ylim([-5, 5])
            axe.scatter(self.train_x, self.train_y, c='blue')

            m = mu.ravel()
            x = np.linspace(-60, 60, self.num_of_test)
            CI = 1.96 * np.sqrt(np.diag(cov))

            axe.plot(x, m, 'red')
            axe.fill_between(x, m+CI, m-CI, facecolor='green', alpha=0.18)
            plt.show()
        
        # plot the result
        plot(mu, cov)
        
    def opt(self):    
        def negative_log_likelihood_loss(params):
            self.alpha, self.l = params[0], params[1]
            C = self.cal_cov(self.train_x)
            neglogli = 0.5 * np.linalg.slogdet(C)[1] + 0.5 * mul(mul(self.train_y.T, inv(C)), self.train_y) + 0.5 * len(self.train_x) * np.log(2*np.pi)
            return neglogli.ravel()
        
        res = minimize(negative_log_likelihood_loss, 
                       [1, 1], 
                       bounds=((1e-5, 1e5), (1e-5, 1e5)))
        
        self.alpha = res.x[0]
        self.l = res.x[1]
        print(self.alpha, self.l)

class SVM:
    def __init__(self, trainX, trainY, testX, testY):
        self.train_x = trainX
        self.train_y = trainY
        self.test_x = testX
        self.test_y = testY
        self.kernel_list = ['0', '1', '2']
        self.kernel_map = ['Linear', 'Polynomial', 'RBF']
        self.task = ['1', '2', '3']
    
    def run(self, task):
        if task == self.task[0]:    # task 1
            with open('SVM_task_1.txt', 'w') as f:
                for k in self.kernel_list:
                    model = svm_train(self.train_y, self.train_x, '-t {}'.format(k))
                    res = svm_predict(self.test_y, self.test_x, model)
                    print("{}".format(res[1][0]), file=f)
  
        if task == self.task[1]:    # task 2
            self.grid_search()
        
        if task == self.task[2]:    # task 3
            train_kernel, test_kernel = self.precomputed_kernel()
            model = svm_train(self.train_y, train_kernel, '-t 4')
            res = svm_predict(self.test_y, test_kernel, model)
            with open('SVM_task_3.txt', 'w') as f:
                print("{}".format(res[1][0]), file=f)

    def grid_search(self):
        C = np.logspace(-5, 15, num=5, base=2, dtype=float)    # cost: 2^-5 ~ 2^15
        gamma = np.logspace(-15, 3, num=5, base=2, dtype=float)    # gamma: 2^-15 ~ 2^3
        degree = [1, 3, 5]    # degree: 1, 3, 5
        coef = [0, 1]   # coef: 0, 1

        opt_param_l, opt_param_p, opt_param_r = '', '', ''
        opt_l, opt_p, opt_r = 0, 0, 0

        f = open('SVM_task_2.txt', 'w')
        
        for k in self.kernel_list:
            for c in C:
                if k == '0':    # linear
                    param = '-t {} -v 3 -c {:f}'.format(k, c)
                    res = svm_train(self.train_y, self.train_x, param)
                    if res > opt_l:
                        opt_l = res
                        opt_param_l = param
                    print("kernel: {{Linear}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                          kernel: {{Polynomial}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                          kernel: {{RBF}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n".format(opt_l, opt_param_l, opt_p, opt_param_p, opt_r, opt_param_r), file=f)

                if k == '1':    # polynomial
                    for g in gamma:
                        for d in degree:
                            for cf in coef:
                                param = '-t {} -v 3 -c {:f} -g {:f} -d {:d} -r {:d}'.format(k, c, g, d, cf)
                                res = svm_train(self.train_y, self.train_x, param)
                                if res > opt_p:
                                    opt_p = res
                                    opt_param_p = param
                                print("kernel: {{Linear}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                                      kernel: {{Polynomial}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                                      kernel: {{RBF}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n".format(opt_l, opt_param_l, opt_p, opt_param_p, opt_r, opt_param_r), file=f)

                if k == '2':    # RBF
                    for g in gamma:
                        param = '-t {} -v 3 -c {:f} -g {:f}'.format(k, c, g)
                        res = svm_train(self.train_y, self.train_x, param)
                        if res > opt_r:
                            opt_r = res
                            opt_param_r = param
                        print("kernel: {{Linear}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                              kernel: {{Polynomial}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n\
                              kernel: {{RBF}}, OPT= {{{:f}}}, OPT_params= {{{:s}}}\n".format(opt_l, opt_param_l, opt_p, opt_param_p, opt_r, opt_param_r), file=f)

        f.close()

    def linera_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def RBF_kernel(self, x1, x2, gamma):
        dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return np.exp(-gamma * dist)

    def precomputed_kernel(self):
        train_kernel = self.linera_kernel(self.train_x, self.train_x) + self.RBF_kernel(self.train_x, self.train_x, 0.015625)
        test_kernel = self.linera_kernel(self.test_x, self.train_x) + self.RBF_kernel(self.test_x, self.train_x, 0.015625)
        train_kernel = np.hstack((np.arange(1, len(self.train_x)+1).reshape(-1, 1), train_kernel))
        test_kernel = np.hstack((np.arange(1, len(self.test_x)+1).reshape(-1, 1), test_kernel))
        return train_kernel, test_kernel

if __name__=='__main__':
    X, Y = load_data("./data/input.data")

    # GPR = GaussianProcess(X, Y)
    # GPR.run()

    train_X = load_csv(("./data/X_train.csv"))
    train_Y = load_csv(("./data/Y_train.csv"))
    test_X = load_csv(("./data/X_test.csv"))
    test_Y = load_csv(("./data/Y_test.csv"))

    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    _SVM = SVM(train_X, train_Y.ravel(), test_X, test_Y.ravel())
    # _SVM.run('1')
    # _SVM.run('2')
    _SVM.run('3')

