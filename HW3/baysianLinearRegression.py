import numpy as np
import math
import matplotlib.pyplot as plt

def univariate_gaussian_data_generator(mean, var):
    z = -6
    for _ in range(12):
        z += np.random.uniform(0, 1)
    return mean + var**(0.5) * z

def polynomial_basis_linear_model_data_generator(n, a, w):
    y = univariate_gaussian_data_generator(0, a)
    x = np.random.uniform(-1, 1)
    for i in range(n):
        y += w[i] * x**(i)
    return x, y

def inv(arr):
    return np.linalg.inv(arr)

def mul(A, B):
    return np.matmul(A, B)

def output(x, y, mean, var, mu, lam):
    print("Add data point (%f, %f):\n" %(x, y))
    print("Posterior mean:")
    for row in mu:
        print("\t%f" %(row.item()))
    
    print("\nPosterior variance:")
    for row in range(len(lam)):
        for column in range(len(lam[row])):
            print("\t%f" %(lam[row][column]), end='')
        print()
    
    print("\nPredictive distribution ~ N(%f, %f)\n" %(mean.item(), var.item()))
    print("==================================================\n")

def gen_design_matrix(x, n):
    X = np.zeros((1, n))
    for i in range(n):
        X[0][i] = x**(i)
    return X

def baysian_linear_regression(b, n, a, w):
    samples = {'10':[], '50':[], 'Final':[]}
    points = []

    lam = np.zeros((n, n))      # initialize lambda (n x n)
    mu = np.zeros((n, 1))       # initialize mu (n x 1)
    pre_mu = np.zeros((n, 1))
    
    for cnt in range(500):
        x, y = polynomial_basis_linear_model_data_generator(n, a, w)
        points.append((x, y))
        X = gen_design_matrix(x, n)

        if cnt == 0:
            lam = (1 / a) * mul(X.T, X) + b * np.identity(n)  # update lambda
            mu = mul((1 / a) * inv(lam), X.T * y) # update mu
        else:
            pre_lam = lam
            pre_mu = mu
            lam = (1 / a) * mul(X.T, X) + pre_lam   # update lambda
            mu = mul(inv(lam), (1 / a) * X.T * y + mul(pre_lam, mu))  # update mu

        # calculate mean and variance
        mean = mul(mu.T, X.T)
        var = a + mul(mul(X, inv(lam)), X.T)

        # output and record information when cnt = 10 and 50
        output(x, y, mean, var, mu, inv(lam))
        if cnt == 10 or cnt == 50:
            samples[str(cnt)].append(mu)
            samples[str(cnt)].append(lam)

        # if converge, then break
        flag = False
        for i in range(len(mu)):
            if(abs(pre_mu[i] - mu[i]) > 1e-12):
                flag = True
        if not flag and cnt > 50:
            break

    samples['Final'].append(mu)
    samples['Final'].append(lam)
    plot(points, n, a, w, samples)

def plot(points, n, a, w, samples):
    num = 500   
    X, Y = [p[0] for p in points], [p[1] for p in points]
    x, y = np.linspace(-2, 2, num), np.zeros(num)
    y_up, y_down = np.zeros(num), np.zeros(num)

    plt.style.use('fast')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    # show ground truth
    for i in range(num):
        for j in range(n):
            y[i] += w[j] * x[i]**(j)
        y_up[i] = y[i] + a
        y_down[i] = y[i] - a
    axes[0][0].set_xlim([-2, 2])
    axes[0][0].set_ylim([-10, 30])
    axes[0][0].set_title("Ground truth")
    axes[0][0].plot(x, y, 'black')
    axes[0][0].plot(x, y_up, 'red')
    axes[0][0].plot(x, y_down, 'red')

    # show predict result
    axes[0][1].set_title("Predict result")
    add_plot(axes[0][1], n, a, samples, 'Final', num)
    axes[0][1].scatter(X, Y)

    # show sample 10 
    axes[1][0].set_title("After 10 incomes")
    add_plot(axes[1][0], n, a, samples, '10', num)
    axes[1][0].scatter(X[0:10], Y[0:10])

    # show sample 50
    axes[1][1].set_title("After 50 incomes")
    add_plot(axes[1][1], n, a, samples, '50', num)
    axes[1][1].scatter(X[0:50], Y[0:50])

    plt.show()

def add_plot(axe, n, a, samples, type, num):
    x, y = np.linspace(-2, 2, num), np.zeros(num)
    y_up, y_down = np.zeros(num), np.zeros(num)
    _mu, _lam = samples[type][0], samples[type][1]

    for i in range(num):
        X = gen_design_matrix(x[i], n)
        y[i] = float(mul(_mu.T, X.T).item())
        y_up[i] = y[i] + float(a + mul(mul(X, inv(_lam)), X.T).item())
        y_down[i] = y[i] - float(a + mul(mul(X, inv(_lam)), X.T).item())

    axe.set_xlim([-2, 2])
    axe.set_ylim([-10, 30])
    axe.plot(x, y, 'black')
    axe.plot(x, y_up, 'red')
    axe.plot(x, y_down, 'red')
        
if __name__ == '__main__':
    b, n, a = map(int, input("Please enter b, n and a:\n").split())
    w = list(map(int, input("Please enter w:\n").split()))

    if len(w) != n:
        print("Error input format for w !")
        quit()

    baysian_linear_regression(b, n, a, w)