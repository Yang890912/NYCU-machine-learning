import math

def Binomial(N, m, p):
    return math.factorial(N) / (math.factorial(N-m) * math.factorial(m)) * p**(m) * (1-p)**(N-m)

def gamma(x):
    if x == 1 or x == 2:
        return 1
    return math.factorial(x-1)

def Beta(a, b, p):
    return gamma(a + b) / (gamma(a) * gamma(b)) * p**(a-1) * (1-p)**(b-1)

if __name__ == '__main__':
    with open('testfile.txt', "r") as input_file:
        a, b = map(int, input("Please enter a and b:\n").split())
        prior, post, idx = 0, 0, 1
        
        for line in input_file.readlines():
            print("Case %d: %s" %(idx, line.rstrip()))

            new_a, new_b = line.count("1"), line.count("0")  # calculate N, m
            print("Likelihood: %.12f" %(Binomial((new_a + new_b), new_a, new_a / (new_a + new_b))))
            print("Beta prior:\t a = %d, b = %d" %(a, b))

            prior = post
            a += new_a  # upgrade alpha
            b += new_b  # upgrade beta
            post = Beta(a, b, a / (a + b))
            print("Beta posterior:  a = %d, b = %d\n" %(a, b))
            
            idx += 1