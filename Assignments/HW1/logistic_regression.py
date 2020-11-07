import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def logistic_calculation():
    sigmoid = [];
    time = [];
    lr = 0.1
    numberOfIteration = 1000
    fit_intercept = True
    verbose = False

    iris = datasets.load_iris()

    # Accessing only 1st two columns of dataset
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    if fit_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        # weights initialization
        theta = np.zeros(X.shape[1])

        for i in range(numberOfIteration):
            # Calculating t so that we can use it in Sigmoid
            t = np.dot(X, theta)
            # Calculating Sigmoid
            # (t) = 1 / 1 + exp^(-t)
            h = 1 / (1 + np.exp(-t))
            # Storing sigmoid values in array so that we can use it for plotting
            sigmoid.append(h)
            # Storing t values in array so that we can use it for plotting
            time.append(t)
            # Calculating gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            # Changing value of theta
            theta -= lr * gradient

            t = np.dot(X, theta)
            h = 1 / (1 + np.exp(-t))
            loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

            if (verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    logistic_function = plt.figure()
    logistic_function.suptitle('Logistic Function', fontweight="bold")
    logistic_function.set_size_inches(8, 8)
    ax = logistic_function.add_subplot(211)
    ax.plot(time, sigmoid)
    plt.xlabel('t')
    plt.show()
    logistic_function.savefig('LogisticFunction.png', dpi=100)


if __name__ == '__main__':
    logistic_calculation()

