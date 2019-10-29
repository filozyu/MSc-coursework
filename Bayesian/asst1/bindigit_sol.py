import matplotlib.pyplot as plt
import numpy as np


def ML_mv_Bernouillie(X):
    q = (1/X.shape[0]) * np.matmul(X.T, np.ones((X.shape[0])))
    return q


def MAP_mv_Bernouillie(X, alpha, beta):
    q = np.matmul(X.T, np.ones((X.shape[0]))) + (alpha - 1) * np.ones((X.shape[1]))
    q /= (X.shape[0] + alpha + beta - 2)
    return q


def main(alpha, beta):
    # load the data set
    Y = np.loadtxt('data/binarydigits.txt')


    plt.figure(figsize=(5, 5))

    q_ml = ML_mv_Bernouillie(Y)
    q_map = MAP_mv_Bernouillie(Y, alpha, beta)

    assert q_ml.shape[0] == Y.shape[1]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(q_ml, (8,8)),
               interpolation="None",
               cmap='gray')
    plt.axis('off')
    plt.title("ML")

    plt.subplot(1, 2, 2)
    plt.imshow(np.reshape(q_map, (8,8)),
               interpolation="None",
               cmap='gray')
    plt.axis('off')
    plt.title("MAP")

    plt.savefig("./mlmap.png")


if __name__ == "__main__":
    alpha = 3
    beta = 3
    main(alpha, beta)
    print("Done!")
