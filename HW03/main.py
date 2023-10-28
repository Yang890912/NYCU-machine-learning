import numpy as np
import matplotlib.pyplot as plt
import math

def univariate_gaussian_data_generator(mean, var):
    """
    ref: https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution

    An easy-to-program approximate approach that relies on the central limit theorem is as follows: 
    1. generate 12 uniform U(0,1) deviates, 
    2. add them all up, 
    3. and subtract 6,
    4. the resulting random variable will have approximately standard normal distribution

    """
    z = -6
    for _ in range(12):
        z += np.random.uniform(0, 1)
    return mean + var**(0.5) * z

def polynomial_basis_linear_model_data_generator(n, a, w):
    """
    Polynomial basis linear model data generator

    x = uniformly distributed (-1 < x < 1)
    e ~ N(0, a)
    y = w_0*x^0 + w_1*x^1 + ... + e

    """
    y = univariate_gaussian_data_generator(0, a)
    x = np.random.uniform(-1, 1)
    for i in range(n):
        y += w[i] * x**(i)
    return x, y

def sequential_estimator(mean, var, iter_rounds):
    """
    Online algorithm to update mean and (estimated) variance

    ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
   
    """
    curr_m = 0
    curr_s = 0
    n = 0
    for _ in range(iter_rounds):
        n += 1
        new_x = univariate_gaussian_data_generator(mean, var)
        print("Add data point: %f" %(new_x))

        # Update mean
        delta = new_x - curr_m
        curr_m += delta / n

        # Update variance
        delta2 = new_x - curr_m
        curr_s += delta * delta2

        print("Mean = %f, Variance = %f\n" %(curr_m, curr_s/n))

def ques_1():
    """
    For Homework question 2

    """
    m, s = map(float, input("Please enter mean and variance:\n").split())
    print("Data point source function: N(%f, %f)" %(m, s))
    sequential_estimator(m, s, 2000)

def design_matrix(x, n):
    """
    X = (1Xn)

    """
    X = np.zeros((1, n))
    for i in range(n):
        X[0][i] = x**(i)
    return X

def inv(arr):
    return np.linalg.inv(arr)

def mul(arr_A, arr_B):
    return np.matmul(arr_A, arr_B)

def output(x, y, mean, var, _mu, _lambda):
    print("Add data point: (%f, %f)\n" %(x, y))
    print("Posterior mean:")
    for row in _mu:
        print("\t%f" %(row))
    
    print("\nPosterior variance:")
    for row in range(len(_lambda)):
        for column in range(len(_lambda[row])):
            print("\t%f" %(_lambda[row][column]), end='')
        print()
    
    print("\nPredictive distribution ~ N(%f, %f)\n" %(mean, var))

def gen_lambda(n, b):
    _lambda = np.zeros((n, n))
    z = univariate_gaussian_data_generator(0, 1/b)
    for i in range(n):
        _lambda[i][i] = z
    return _lambda

def Baysian_linear_regression(b, n, a, w):
    _lambda = gen_lambda(n, b)    # initial lambda    (nXn)
    _mu = np.zeros((n, 1))        # initial mu        (nX1)
    samples = {'10':[], '50':[], 'final':[]}
    data_points = []
    a = 1/a                 # the fornmula denotes a^-1 as var 
    _lambda = inv(_lambda)  # the fornmula denotes ^^-1 as covar 

    for cnt in range(200):
        x, y = polynomial_basis_linear_model_data_generator(n, 1/a, w)
        data_points.append((x, y))
        X = design_matrix(x, n)

        if cnt == 0:
            # Update lambda
            _lambda = a * mul(X.T, X) + b * np.identity(n)
            
            # Update mu
            m = _mu
            t1 = a * y * X.T
            _mu = mul(inv(_lambda), t1)
        else:                   
            # Update lambda
            C = _lambda
            _lambda = a * mul(X.T, X) + C
            
            # Update mu
            m = _mu
            t1 = a * y * X.T
            t2 = mul(C, m)
            _mu = mul(inv(_lambda), t1+t2)

        # Calculate mean, var
        mean = mul(_mu.T, X.T)
        var = 1/a + mul(mul(X, inv(_lambda)), X.T)

        # Output and record info at cnt = 10 and 50
        output(x, y, mean, var, _mu, inv(_lambda))
        if cnt == 10 or cnt == 50:
            samples[str(cnt)].append(_mu)
            samples[str(cnt)].append(_lambda)

        # If converge, break
        flag = False
        for i in range(len(_mu)):
            if(abs(m[i]-_mu[i]) > 1e-4):
                flag = True
        if not flag and cnt > 50:
            break

    samples['final'].append(_mu)
    samples['final'].append(_lambda)
    plot(data_points, n, a, w, samples)

def ques_2():
    """
    For Homework question 3

    """
    b, n, a = map(int, input("Please enter b, n and a:\n").split())
    w = list(map(int, input("Please enter w:\n").split()))

    if len(w) != n:
        print("Error input format for w !")
        quit()

    Baysian_linear_regression(b, n, a, w)

def plot(points, n, a, w, samples):
    """
    Visualization 

    """
    p_num = 500
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    x = np.linspace(-2, 2, p_num)
    y = np.zeros(p_num)
    yu = np.zeros(p_num)
    yd = np.zeros(p_num)

    plt.style.use('fast')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    # Ground truth
    for i in range(p_num):
        for j in range(n):
            y[i] += w[j] * x[i]**(j)
        yu[i] = y[i] + 1/a
        yd[i] = y[i] - 1/a
    axes[0][0].set_xlim([-2, 2])
    axes[0][0].set_ylim([-10, 30])
    axes[0][0].set_title("Ground truth")
    axes[0][0].plot(x, y, 'black')
    axes[0][0].plot(x, yu, 'red')
    axes[0][0].plot(x, yd, 'red')

    # Predict result
    axes[0][1].set_title("Predict result")
    add_plot(axes[0][1], n, a, samples, 'final')
    axes[0][1].scatter(X, Y)

    # Sample 10 
    axes[1][0].set_title("After 10 incomes")
    add_plot(axes[1][0], n, a, samples, '10')
    axes[1][0].scatter(X[0:10], Y[0:10])

    # Sample 50
    axes[1][1].set_title("After 50 incomes")
    add_plot(axes[1][1], n, a, samples, '50')
    axes[1][1].scatter(X[0:50], Y[0:50])

    plt.show()

def add_plot(axe, n, a, samples, type):
    p_num = 500
    x = np.linspace(-2, 2, p_num)
    y = np.zeros(p_num)
    yu = np.zeros(p_num)
    yd = np.zeros(p_num)
    _mu = samples[type][0]
    _lambda = samples[type][1]

    for i in range(p_num):
        X = design_matrix(x[i], n)
        y[i] = float(mul(_mu.T, X.T))
        yu[i] = y[i] + float(1/a + mul(mul(X, inv(_lambda)), X.T))
        yd[i] = y[i] - float(1/a + mul(mul(X, inv(_lambda)), X.T))

    axe.set_xlim([-2, 2])
    axe.set_ylim([-10, 30])
    axe.plot(x, y, 'black')
    axe.plot(x, yu, 'red')
    axe.plot(x, yd, 'red')
    
    
if __name__ == '__main__':
    # ques_1()
    ques_2()


 



