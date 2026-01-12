from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from resolution_functions.instrument import Instrument


DATA_PATH = Path(__file__).parent / 'data' / 'ideal'

def _get_boxcar_kernel():
    points = np.array([[1.], [2.]])
    mesh = np.linspace(-5, 5, 11)
    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('boxcar', width=3.)

    return model.get_kernel(points, mesh), mesh

def test_boxcar_kernel():
    result, _ = _get_boxcar_kernel()
    assert_allclose(result, np.load(DATA_PATH / '_get_boxcar_kernel.npy'))

def _get_boxcar_peak():
    points = np.arange(0, 4, 2)[:, None]
    mesh = np.linspace(-5, 5, 11)
    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('boxcar', width=3.)

    return model.get_peak(points, mesh), mesh

def test_boxcar_peak():
    result, _ = _get_boxcar_peak()
    assert_allclose(result, np.load(DATA_PATH / '_get_boxcar_peak.npy'))

def _get_boxcar_broaden():
    mesh = np.linspace(0, 5, 11)  # i.e. each bin is 0.5
    points = mesh[:, None]
    data = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  # i.e. pulse area = 1

    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('boxcar', width=1.)

    return model.broaden(points, data, mesh), mesh

def test_boxcar_broaden():
    result, _ = _get_boxcar_broaden()
    assert_allclose(result, np.load(DATA_PATH / '_get_boxcar_broaden.npy'))

def _get_triangle_kernel():
    points = np.array([[3.], [7.]])
    mesh = np.arange(-5., 5., 1.)
    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('triangle', fwhm=2.)

    return model.get_kernel(points, mesh), mesh

def test_triangle_kernel():
    result, _ = _get_triangle_kernel()
    assert_allclose(result, np.load(DATA_PATH / '_get_triangle_kernel.npy'))


def _get_triangle_peak():
    points = np.array([[3.], [7.]])
    mesh = np.arange(0., 10., 1.)

    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('triangle', fwhm=2.)

    return model.get_peak(points, mesh), mesh

def test_triangle_peak():
    result, _ = _get_triangle_peak()
    assert_allclose(result, np.load(DATA_PATH / '_get_triangle_peak.npy'))

def _get_triangle_broaden():
    points = np.array([[3.], [7.]])
    mesh = np.arange(0., 10., 1.)
    data = [0.5, 2.]
    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function('triangle', fwhm=2.)

    return model.broaden(points, data, mesh), mesh

def test_triangle_broaden():
    result, _ = _get_triangle_broaden()
    assert_allclose(result, np.load(DATA_PATH / '_get_triangle_broaden.npy'))


_GET_DATA_FUNCTIONS = [
    _get_boxcar_kernel,
    _get_boxcar_peak,
    _get_boxcar_broaden,
    _get_triangle_kernel,
    _get_triangle_peak,
    _get_triangle_broaden,
]


def generate_data():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    DATA_PATH.mkdir(exist_ok=True)

    for func in _GET_DATA_FUNCTIONS:
        name = str(func.__name__)
        print(name)

        result, mesh = func()
        np.save(DATA_PATH / f'{name}.npy', result)

        fig, ax = plt.subplots()
        if len(np.shape(result)) == 1:
            ax.plot(mesh, result)
        else:
            for row in result:
                ax.plot(mesh, row)

        fig.savefig(DATA_PATH / f'{name}.png')
        plt.close(fig)


if __name__ == '__main__':
    generate_data()
