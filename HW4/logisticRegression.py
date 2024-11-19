import numpy as np
import matplotlib.pyplot as plt
import math

def univariate_gaussian_data_generator(mean, var):
    z = -6
    for _ in range(12):
        z += np.random.uniform(0, 1)
    return mean + var**(0.5) * z

def gen_points(num_of_points, mx, vx, my, vy):
    points = []
    for _ in range(num_of_points):
        x = univariate_gaussian_data_generator(mx, vx)
        y = univariate_gaussian_data_generator(my, vy)
        points.append((x, y))
    return points

def gen_design_matrix(D1, D2):
    A = np.zeros((len(D1) + len(D2), 3))
    y = np.zeros((len(D1) + len(D2), 1))  # D1 => 0, D2 => 1
    idx = 0
    for p in D1:
        A[idx][0], A[idx][1], A[idx][2] = 1, p[0], p[1]
        idx += 1
    for p in D2:
        A[idx][0], A[idx][1], A[idx][2] = 1, p[0], p[1]
        y[idx][0] = 1
        idx += 1
    return A, y

def sigmoid_fun(x):
    return 1 / (1 + math.exp(-1 * x))

def mul(A, B):
    return np.matmul(A, B)

def inv(arr):
    return np.linalg.inv(arr)

def gradient_descent(A, y, lr=0.02):
    w = np.zeros((3, 1))    # initialize w = [0 0 0]^T
    n = len(y)
    step = 0
    
    while True:
        step += 1
        g = mul(A, w)
        for i in range(n):
            g[i] = sigmoid_fun(g[i].item()) - y[i]
        g = mul(A.T, g)
        w_new = w - lr * g

        if sum(abs(w_new - w)) < 1e-4 or step > 5000:   # if converge or times of step > 5000, then break
            break 

        w = w_new   # update w
        
    pred = output(A, w, y, "Gradient descent")
    return pred

def newton_method(A, y, lr=0.02):
    w = np.zeros((3, 1))    # initialize w = [0 0 0]^T
    n = len(y)
    step = 0

    while True:
        step += 1
        g = mul(A, w)
        for i in range(n):
            g[i] = sigmoid_fun(g[i].item()) - y[i]
        g = mul(A.T, g)

        H = np.zeros((n, n))
        for i in range(n):
            H[i][i] = math.exp(-1 * mul(A[i], w).item()) / (1 + math.exp(-1 * mul(A[i], w).item()))**2
        H = mul(mul(A.T, H), A)
        H_inv = np.identity(n) if np.linalg.det(H) == 0 else inv(H) # if H cannot inverse, then use gradient
        w_new = w - lr * mul(H_inv, g)

        if sum(abs(w_new - w)) < 1e-4 or step > 5000:   # if converge or times of step > 5000, then break
            break

        w = w_new   # update w
        
    pred = output(A, w, y, "Newton's method")
    return pred

def estimate_result(A, w, y):
    TP, FN, FP, TN = 0, 0, 0, 0
    pred = mul(A, w)
    for i in range(len(pred)):
        pred[i] = 1 if sigmoid_fun(pred[i].item()) >= 0.5 else 0
        TP += 1 if y[i] == 0 and pred[i] == 0 else 0
        FN += 1 if y[i] == 0 and pred[i] == 1 else 0
        FP += 1 if y[i] == 1 and pred[i] == 0 else 0
        TN += 1 if y[i] == 1 and pred[i] == 1 else 0
    return pred, TP, FN, FP, TN

def output(A, w, y, method):
    print("\n{:s}:\n".format(method))
    print("w:")

    for w_i in w:
        print("\t%f" %w_i.item())
    
    print("\nConfusion Matrix:\n")
    print("\t\tPredict cluster 1 \t Predict cluster 2")

    pred, TP, FN, FP, TN = estimate_result(A, w, y)
    print("Is cluster 1 \t\t{:^2d}\t\t\t{:^2d}".format(TP, FN))
    print("Is cluster 2 \t\t{:^2d}\t\t\t{:^2d}".format(FP, TN))
    print("\nSensitivity (Successfully predict cluster 1): {:<f}".format(TP/(TP+FN)))
    print("Specificity (Successfully predict cluster 2): {:<f}\n".format(TN/(FP+TN)))
    print("==========================================================")
    return pred

def add_plot(axe, y, method):
    rX, rY, bX, bY = [], [], [], []
    for i in range(len(y)):
        if y[i] == 0:
            rX.append(A[i][1])
            rY.append(A[i][2])
        else:
            bX.append(A[i][1])
            bY.append(A[i][2])
    axe.set_xlim([-5, 15])
    axe.set_ylim([-10, 30])
    axe.set_title(method)
    axe.scatter(rX, rY, c='red')
    axe.scatter(bX, bY, c='blue')

def plot(y, pred_g, pred_n):
    plt.style.use('fast')
    fig, axes = plt.subplots(1, 3, figsize=(4, 10), constrained_layout=True)

    add_plot(axes[0], y, "Ground truth")    # ground truth
    add_plot(axes[1], pred_g, "Gradient descent")   # gradient descent
    add_plot(axes[2], pred_n, "Newton's method")    # newton's method

    plt.show()

if __name__ == '__main__':
    N = int(input("Please enter the number of points:\n"))
    mx1, my1, mx2, my2, vx1, vy1, vx2, vy2 = map(int, input("Please enter mx1, my1, mx2, my2, vx1, vy1, vx2, vy2:\n").split())

    D1 = gen_points(N, mx1, vx1, my1, vy1)
    D2 = gen_points(N, mx2, vx2, my2, vy2)
    A, y = gen_design_matrix(D1, D2)

    lr = 0.02 # learning rate
    pred_g = gradient_descent(A, y, lr)
    pred_n = newton_method(A, y, lr)
    plot(y, pred_g, pred_n)

# 50 1 1 10 10 2 2 2 2
# 50 1 1 3 3 2 2 4 4