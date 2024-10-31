import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


class LinearKalmanFilter:
    def __init__(self, dim_x, A, H, Q, R, x0):
        self.dim_x = dim_x
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = np.eye(dim_x)

    def predict(self):
        # Project the state ahead
        self.x_ = self.A @ self.x

        # Project the covariance ahead
        self.P_ = self.A @ self.P @ self.A.T + self.Q

        return self.x_

    def correct(self, z):
        # Compute the Kalman gain
        K = self.P_ @ self.H.T @ np.linalg.inv(self.H @ self.P_ @ self.H.T + self.R)

        # Update state estimate
        self.x = self.x_ + K @ (z - self.H @ self.x_)

        # Update the covariance
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P_

        return self.x


if __name__ == '__main__':

    # Simulation
    N = 100
    truths = [n * 2 + 10 for n in range(N)]
    sensing_std = 300 ** 0.5
    measures = [t + randn() * sensing_std for t in truths]

    kf = LinearKalmanFilter(
        dim_x=2,
        A=np.matrix([[1, 1], [0, 1]]),
        H=np.matrix([[1, 0]]),
        Q=np.eye(2) * 0.005,
        R=np.eye(1) * sensing_std,
        x0=np.matrix([[0], [0]])
    )

    esti = []
    for z in measures:
        kf.predict()
        x_hat = kf.correct(z)
        esti.append(x_hat[0, 0])

    plt.plot(truths, 'g+')
    plt.plot(measures, 'b+')
    plt.plot(esti, 'r-')
    plt.show()
