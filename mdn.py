import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.distributions import Normal, StudentT
from torch.autograd import Variable

# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(nn.Linear(1, n_hidden),
                                 nn.Tanh())
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu


def mdn_loss_fn(pi, sigma, mu, y):
    """
    -log(sum_k(pi_k*N(y, mu_k(x), sigma_k(x))))
    """
    result = pi*torch.exp(Normal(mu, sigma).log_prob(y))
    # result = pi*torch.exp(StudentT(torch.tensor([2.0]), mu, sigma).log_prob(y))
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


def gumbel_sample(x, axis=1):
    """
    http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    """
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)


def fit(x_data, y_data, x_test_data=None, n_train=3000, n_hidden=10,
        n_gaussians=2):
    n_input = 1
    n_samples = len(x_data)
    mdn_x_tensor = torch.from_numpy(np.float32(x_data).reshape(n_samples, n_input))
    mdn_y_tensor = torch.from_numpy(np.float32(y_data).reshape(n_samples, n_input))
    x_variable = Variable(mdn_x_tensor)
    y_variable = Variable(mdn_y_tensor, requires_grad=False)

    if x_test_data is None:
        x_test_data = np.linspace(np.min(x_data), np.max(x_data), n_samples)

    x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
    x_test_variable = Variable(x_test_tensor)

    network = MDN(n_hidden=n_hidden, n_gaussians=n_gaussians)
    optimizer = torch.optim.Adam(network.parameters())

    for epoch in range(n_train):
        pi_variable, sigma_variable, mu_variable = network(x_variable)
        loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_variable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%500 == 0:
            print(epoch, loss.data[0])

    pi_variable, sigma_variable, mu_variable = network(x_test_variable)

    pi_data = pi_variable.data.numpy()
    sigma_data = sigma_variable.data.numpy()
    mu_data = mu_variable.data.numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax1.plot(x_test_data, pi_data)
    ax1.set_title('$\Pi$')
    ax2.plot(x_test_data, sigma_data)
    ax2.set_title('$\sigma$')
    ax3.plot(x_test_data, mu_data)
    ax3.set_title('$\mu$')
    plt.show()

    plt.figure(figsize=(8, 8), facecolor='white')
    for mu_k, sigma_k in zip(mu_data.T, sigma_data.T):
        plt.plot(x_test_data, mu_k)
        plt.fill_between(x_test_data, mu_k-sigma_k, mu_k+sigma_k, alpha=0.1)
    plt.scatter(x_data, y_data, marker='.', lw=0, alpha=0.2, c='black')
    plt.show()

    k = gumbel_sample(pi_data)
    indices = (np.arange(n_samples), k)
    rn = np.random.randn(n_samples)
    sampled = rn*sigma_data[indices]+mu_data[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.2)
    plt.scatter(x_test_data, sampled, alpha=0.2, color='red')
    plt.show()

    return pi_data, mu_data, sigma_data


if __name__ == "__main__":

    # Generate and plot artificial data
    def generate_data(n_samples, mu1=0, mu2=1):
        x_data = stats.truncnorm.rvs(0, 2, loc=0, scale=5, size=n_samples)
        y_data1 = np.random.normal(loc=mu1,
                                   scale=0.05*x_data[:n_samples/2],
                                   size=n_samples/2)
        y_data2 = np.random.normal(loc=mu2,
                                   scale=0.05*x_data[n_samples/2:],
                                   size=n_samples/2)
        return x_data, np.concatenate((y_data1, y_data2))


    n_samples = 2000
    x_data, y_data = generate_data(n_samples)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.2)
    plt.show()

    pi_data, mu_data, sigma_data = fit(x_data, y_data)