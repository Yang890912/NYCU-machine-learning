import numpy as np
import math

def univariate_gaussian_data_generator(mean, var):
    """
    An easy-to-program approximate approach that relies on the central limit theorem is as follows: 

        1. generate 12 uniform U(0,1) deviates, 
        2. add them all up, 
        3. subtract 6,
        4. the resulting random variable will have approximately standard normal distribution

    Ref: https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution

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
        y = w_0 * x^0 + w_1 * x^1 + ... + e

    """
    y = univariate_gaussian_data_generator(0, a)
    x = np.random.uniform(-1, 1)
    for i in range(n):
        y += w[i] * x**(i)
    return x, y

def sequential_estimator(mean, var, num_of_rounds):
    """
    Online algorithm to update mean and (estimated) variance

    Ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    """
    curr_m, curr_s, n = 0, 0, 0
    for _ in range(num_of_rounds):
        n += 1
        new_x = univariate_gaussian_data_generator(mean, var)
        print("Add data point: %f" %(new_x))

        # update mean
        delta = new_x - curr_m
        curr_m += delta / n

        # update variance
        delta2 = new_x - curr_m
        curr_s += delta * delta2

        print("Mean = %f, Variance = %f\n" %(curr_m, curr_s / n))
        print("====================================\n")

if __name__ == '__main__':
    m, s = map(float, input("Please enter mean and variance:\n").split())
    print("Data point source function: N(%f, %f)" %(m, s))

    sequential_estimator(m, s, 5000)