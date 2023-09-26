import sys
sys.path.append("..")
import matplotlib.pyplot as plt   
import numpy as np
from LSE import LSE
from Newton import Newton
from SteepestDescent import SteepestDescent

def load_points():
    print("Please enter file path:")
    path = input()
    f = open(path, 'r')
    points = []
    for line in f.readlines():
        pair = line.split(',')
        points.append([float(pair[0]), float(pair[1])]) 
    f.close()
    return points

def gen_fitting_line(c):
    s = str()
    for i, _c in enumerate(c):
        if i == 0:
            s = '%fX^%d' %(_c, len(c)-1-i)
        elif i == len(c)-1:
            s += '%s %f' %(" +" if _c < 0 else " -", abs(_c))
        else:
            s += '%s %fX^%d' %(" +" if _c < 0 else " -", abs(_c), len(c)-1-i)  
    return s

def gen_total_error(c, points):   # RSS
    E = 0
    for point in points:
        x, y, sum = point[0], point[1], 0
        for i, _c in enumerate(c):
            sum += _c*(x**(len(c)-1-i))
        E += (y-sum)**2
    return E

def visualization(results, points):
    newp = (np.asarray(points).T).tolist()
    xmin, xmax = min(newp[0]), max(newp[0])
    x = np.arange(xmin, xmax, 0.1)

    # LSE
    plt.subplot(3, 1, 1) 
    plt.scatter(newp[0], newp[1], color = "red")
    plt.title("LSE", {'fontsize':12})
    r = results[0]
    y = sum([c*x**(len(r)-1-i) for i, c in enumerate(r)])
    plt.plot(x, y, color = "black")

    # Steepest descent method
    plt.subplot(3, 1, 2) 
    plt.scatter(newp[0], newp[1], color = "red")
    plt.title("Steepest Descent Method", {'fontsize':12})
    r = results[1]
    y = sum([c*x**(len(r)-1-i) for i, c in enumerate(r)])
    plt.plot(x, y, color = "black")

    # Newton's method
    plt.subplot(3, 1, 3) 
    plt.scatter(newp[0], newp[1], color = "red")
    plt.title("Newton's Method", {'fontsize':12})
    r = results[2]
    y = sum([c*x**(len(r)-1-i) for i, c in enumerate(r)])
    plt.plot(x, y, color = "black")

    plt.show()

if __name__ == '__main__':
    points = load_points()
    print("Please enter n:")
    n = int(input())
    print("Please enter lambda:")
    l = int(input())
    lr = 0.000005 # learning rate 

    m1 = LSE(n, l, points)
    m2 = SteepestDescent(n, lr, points)
    m3 = Newton(n, points)
    methods = [m1, m2, m3]
    results = []

    for m in methods:
        result = m.run()
        results.append(result)
        print("\n%s:" %(m.name))
        print("Fitting line: %s" %(gen_fitting_line(result.ravel().tolist())))
        print("Total error: %f" %(gen_total_error(result.ravel().tolist(), points)))
        
    visualization(results, points)
    

