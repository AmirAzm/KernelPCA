import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd


class K_PCA(object):
    def __init__(self, k, kernel):
        self.k = k
        self.kernel = kernel
        if self.kernel == 'RBF':
            self.fun = self.RBF
        elif self.kernel == 'polynomial':
            self.fun = self.polynomial
        return

    def fit_transform(self, x):
        n = x.shape[0]
        self.k_matrix = np.zeros((n, n))
        for (i, j), val in np.ndenumerate(self.k_matrix):
            self.k_matrix[i, j] += self.fun(x[i], x[j])
        J = 1 / n * np.ones((n, n))
        C = self.k_matrix - (np.dot(self.k_matrix, J)) - (np.dot(J, self.k_matrix)) - (np.dot(J, np.dot(self.k_matrix, J)))
        eigval, eigvec = np.linalg.eig(C)
        idx = eigval.argsort()[::-1]
        eigvec = eigvec[:, idx]
        eigval = eigval[idx]
        x_p = eigvec[:, :self.k]
        return x_p

    def polynomial(self, x, y, order=5):
        return (np.dot(x.T, y))**order

    def RBF(self, x, y, gamma=5):
        return np.exp(-gamma * (np.linalg.norm(x - y))**2)


def sample_sphere(n_sample, n_class, R, plot=True):
    data = pd.DataFrame(columns=['x', 'y', 'z', 'class'])
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    n = n_sample // n_class
    for index, r in enumerate(R):
        phi = np.random.rand(n) * np.pi
        theta = np.random.rand(n) * 2 * np.pi
        x, y, z = sphere_to_cart(r, phi, theta)
        c = np.ones(n) * index
        data = data.append(pd.DataFrame(np.array([x, y, z, c]).T, columns=['x', 'y', 'z', 'class']), ignore_index=True)
        ax.scatter3D(x, y, z, label=index, cmap='GnBu')
        ax.legend()
    if plot:
        plt.show()
    return data


def sphere_to_cart(r, phi, theta):
    x = r * (np.sin(theta) * np.cos(phi))
    y = r * (np.sin(theta) * np.sin(phi))
    z = r * (np.cos(theta))
    return x, y, z


dataset = sample_sphere(1000, 2, [1, 3], plot=True)
data = dataset.iloc[:, :3].values
target = dataset.iloc[:, 3].values
k_pca = K_PCA(k=2, kernel='polynomial')
data_new = k_pca.fit_transform(data)
fig1 = plt.figure(figsize=(10, 7))
plt.scatter(data_new[:, 0], data_new[:, 1], c=target[:])
fig2 = plt.figure(figsize=(10, 7))
plt.scatter(data_new[:, 0], np.zeros(data_new.shape[0]), c=target[:])
plt.show()
