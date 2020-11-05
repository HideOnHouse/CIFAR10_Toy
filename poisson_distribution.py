from matplotlib import pyplot as plt


def factorial(x):
    result = 1
    for i in range(1, x + 1):
        result *= i
    return result


def poisson(lamb, n):
    return (lamb ** n) * (2.7182818284 ** (-lamb)) / factorial(n)


def main():
    k = list(range(20))
    for i in range(1, 5):
        plt.plot(k, list(map(poisson, list(i for _ in range(len(k))), k)), label='lambda: ' + str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
