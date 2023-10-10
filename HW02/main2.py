import math

def Binomial(N, m, p):
    return math.factorial(N) // (math.factorial(N-m)*math.factorial(m)) * p**(m) * (1-p)**(N-m)

def r(x):
    if x == 1 or x == 2:
        return 1
    return math.factorial(x-1)

def Beta_dis(a, b, p):
    return r(a+b) / (r(a)*r(b)) * p**(a-1) * (1-p)**(b-1)

if __name__ == '__main__':

    with open('testfile.txt', "r") as input_file:
        a, b = map(int, input("Please enter a and b:\n").split())
        prior, post = 0, 0
        idx = 1
        for line in input_file.readlines():
            print("Case %d: %s" %(idx, line.rstrip()))

            # Calculate N, m
            aNew = line.count("1")
            bNew = line.count("0")
            print("Likelihood: %.12f" %(Binomial(aNew+bNew, aNew, aNew/(aNew+bNew))))
            print("Beta prior:\t a = %d, b = %d" %(a, b))
            prior = post

            # Updare alpha, beta
            a += aNew
            b += bNew
            print("Beta posterior:  a = %d, b = %d\n" %(a, b))
            # print(a/(a+b))
            post = Beta_dis(a, b, a/(a+b))
            # print(post)

            idx += 1