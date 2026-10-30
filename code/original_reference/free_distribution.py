import numpy as np
import scipy.integrate as integrate


def My_distribution(f):
    # Calculate the cumulative distribution function F(x)
    def F(x):
        return integrate.quad(f, -np.inf, x)[0]

    # Inverse transform sampling
    def sample_from_distribution():
        U = np.random.rand()  # Generate a random number U ~ Uniform(0, 1)
        # Find X such that F(X) = U using numerical methods (e.g., bisection method)
        # Here we use scipy.optimize.brentq to find the root of F(X) - U = 0
        # X = U
        # while abs(F(X)-U)>0.01:
        #     if F(X)>U:
        #         X -= 0.001
        #     else:
        #         X += 0.001
        #     # print(X)
        return 1/(1-U)**(1/3)

    return sample_from_distribution


# Example usage:
# Define your probability density function f(x)
def f(x):
    x0 = 1
    lambd = 3
    if x >= x0:
        r = (x/x0)**(1-lambd)
    if 0 <= x < x0:
        r = 1
    if x < 0:
        r = 0
    r /= x0*(lambd-1)/(lambd-2)
    return r

if __name__ == '__main__':
    # Create a function to sample from the distribution defined by f(x)
    sample = My_distribution(f)

    # Generate a sample from the distribution
    # generated_value = sample()
    # print("Generated value from the distribution:", generated_value)
    from plot_distribution import plot_distribution as pd
    data = []
    for _ in range(1000):
        data.append(sample())
        print(_)
    pd(data, 100)
