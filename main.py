import tensorflow as tf
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from models.model_trainer import ModelTrainer
from models.multi_fidelity_deep_gp import MultiFidelityDeepGPTrainer as DeepTrainer

import os


def fh1(x):
    return np.power(6 * x - 2, 2) * np.sin(12 * x - 4) + 10


def fh2(x):
    return 1.5 * (x + 2.5) * np.sqrt(fh1(x))


def fh3(x):
    return 5 * x * x * np.sin(12 ** x)


def fl1(x):
    return 2 * (x + 2) * np.sqrt(fh1(x) - 1.3 * (np.power(6 * x - 2, 2) - 6 * x))


def fl2(x):
    return np.power(6 * x - 2, 2) * np.sin(8 * x - 4) + 10 - (np.power(6 * x - 2, 2) - 6 * x)


def fl3(x):
    return fh3(x) + (x * x * x * np.sin(3 * x - 0.5)) + 4 * np.cos(2 * x)


fh = [fh1, fh2, fh3]
fl = [fl1, fl2, fl3]





def plot_gp(x, mu, var, color, label, ax=None):
    if ax:
        ax.plot(x, mu, color=color, lw=2, label=label)
        ax.fill_between(x[:, 0],
                        (mu - 2 * np.sqrt(np.abs(var)))[:, 0],
                        (mu + 2 * np.sqrt(np.abs(var)))[:, 0],
                        color=color, alpha=0.4)
    else:
        plt.plot(x, mu, color=color, lw=2, label=label)
        plt.fill_between(x[:, 0],
                        (mu - 2 * np.sqrt(np.abs(var)))[:, 0],
                        (mu + 2 * np.sqrt(np.abs(var)))[:, 0],
                        color=color, alpha=0.4)


def plot(m, m2=None):
    xtest = np.linspace(0, 1, 100)[:, None]
    
    f, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes = axes

    for i in range(len(axes)):
        axes[i].plot(xtest, fh[i](xtest), label="Y")
        mu, var = m.predict(np.hstack((xtest, np.ones_like(xtest)*i)))
        axes[i].scatter(sample_hf, fh[i](sample_hf), label = "HF Training samples")
        plot_gp(xtest, mu, var,'tab:orange' , label=f"Predicted Y, {num_lo_fi} LF samples", ax=axes[i])
        if m2:
            mu, var =  m2.predict(np.hstack((xtest, np.ones_like(xtest)*i)))
            plot_gp(xtest, mu, var,'tab:green' , label=f"Predicted Y, {num_lo_fi2} LF samples", ax=axes[i])

    axes[1].legend()
    f.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_lo_fi = 10
    num_hi_fi = 6

    sampler = qmc.LatinHypercube(d=1)
    sample_hf = sampler.random(n=num_hi_fi).transpose()[0]
    sample_lf = sampler.random(n=num_lo_fi).transpose()[0]

    xl = sample_lf.reshape(sample_lf.shape[0], 1)
    xh = sample_hf.reshape(sample_hf.shape[0], 1)

    X = np.vstack((
            np.hstack((xl, np.zeros_like(xl), np.zeros_like(xl))),
            np.hstack((xl, np.ones_like(xl), np.zeros_like(xl))),
            np.hstack((xl, np.ones_like(xl) * 2, np.zeros_like(xl))),
            np.hstack((xh, np.zeros_like(xh), np.ones_like(xh))),
            np.hstack((xh, np.ones_like(xh), np.ones_like(xh))),
            np.hstack((xh, np.ones_like(xh) * 2, np.ones_like(xh)))
    ))

    Y = np.vstack((
            np.hstack((fl1(xl), np.zeros_like(xl), np.zeros_like(xl))),
            np.hstack((fl2(xl), np.ones_like(xl), np.zeros_like(xl))),
            np.hstack((fl3(xl), np.ones_like(xl) * 2, np.zeros_like(xl))),
            np.hstack((fh1(xh), np.zeros_like(xh), np.ones_like(xh))),
            np.hstack((fh2(xh), np.ones_like(xh), np.ones_like(xh))),
            np.hstack((fh3(xh), np.ones_like(xh) * 2, np.ones_like(xh)))
    ))

    model_name = 'DGP'
    base_kernel = 'RBF'
    likelihood_name = 'Gaussian'

    # trainer = ModelTrainer(
    #     data=(X, Y),
    #     optimizer_name='scipy',
    #     num_outputs=3
    # )
    # trainer.construct_model(
    #     model_names=model_name,
    #     base_kernel=base_kernel,
    #     likelihood_name=likelihood_name
    # )
    # trainer.train_model()

    deep_trainer = DeepTrainer(
        data=(X, Y),
        optimizer_name='scipy',
        num_outputs=3
    )

    deep_trainer.construct_model(
        model_names=['VGP', 'VGP'],
        base_kernel=base_kernel,
        likelihood_name=likelihood_name
    )

    deep_trainer.train_deep_model()


    # trainer2 = ModelTrainer(
    #     model_name=model_name,
    #     optimizer_name='scipy',
    #     kernel_names=[base_kernel, 'Coregion'],
    #     likelihood_name=likelihood_name,
    #     X=X2,
    #     Y=Y2,
    #     num_outputs=2
    # )
    # trainer2.train_model()
    plot(deep_trainer)

