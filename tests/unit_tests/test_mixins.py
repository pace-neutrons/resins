import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

from resolution_functions.models.mixins import GaussianKernel1DMixin, SimpleConvolve1DMixin

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class MockModelLinearWidth(GaussianKernel1DMixin):
    def get_characteristics(self, omega_q):
        return {'sigma': np.arange(1, len(omega_q)+1)}


class MockModel(SimpleConvolve1DMixin):
    def __init__(self, idxs: list[int]):
        self.idxs = idxs

    def get_peak(self, omega_q, mesh):
        mesh_len = len(mesh)
        omega_len = len(omega_q)

        new_mesh = np.zeros((omega_len, mesh_len))
        for i, idx in enumerate(self.idxs):
            new_mesh[i, np.arange(idx-5, idx+5, 1)] = 0.5

        return new_mesh


def _get_kernel_gaussian1d():
    omega_q = np.arange(0, 2000, 50)[:, np.newaxis]
    mesh = np.linspace(-100, 100, 1000)
    model = MockModelLinearWidth()

    return model.get_kernel(omega_q, mesh), mesh


def test_get_kernel():
    result, _ = _get_kernel_gaussian1d()
    assert_allclose(result, np.load(os.path.join(DATA_DIR, '_get_kernel_gaussian1d.npy')))


def _get_peak_gaussian1d():
    omega_q = np.arange(0, 2000, 50)[:, np.newaxis]
    mesh = np.linspace(-100, 2100, 10000)
    model = MockModelLinearWidth()

    return model.get_peak(omega_q, mesh), mesh


def test_get_peak_gaussian1d():
    result, _ = _get_peak_gaussian1d()
    assert_allclose(result, np.load(os.path.join(DATA_DIR, '_get_peak_gaussian1d.npy')))


def _convolve_simple():
    omega_q = np.arange(0, 2000, 50)[:, np.newaxis]
    mesh = np.arange(-100, 2100, 0.25)

    np.random.seed(42)
    data = mesh.copy()
    idxs = np.arange(400, 8400, 200)
    data[idxs] = np.random.random(40)
    model = MockModel(idxs)

    return model.convolve(omega_q, data, mesh), mesh


def test_convolve_simple():
    result, _ = _convolve_simple()
    assert_allclose(result, np.load(os.path.join(DATA_DIR, '_convolve_simple.npy')))


def generate_data():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    for func in [_convolve_simple]:#[_get_kernel_gaussian1d, _get_peak_gaussian1d, _convolve_simple]:
        name = str(func.__name__)
        print(name)

        result, mesh = func()
        np.save(os.path.join(DATA_DIR, name + '.npy'), result)

        fig, ax = plt.subplots()
        if len(np.shape(result)) == 1:
            ax.plot(mesh, result)
        else:
            for row in result:
                ax.plot(mesh, row)

        fig.savefig(os.path.join(DATA_DIR, name + '.png'))
        plt.close(fig)


if __name__ == '__main__':
    generate_data()