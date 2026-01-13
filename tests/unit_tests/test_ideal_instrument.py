from collections import ChainMap
from enum import StrEnum
from itertools import product
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from resolution_functions.instrument import Instrument


DATA_PATH = Path(__file__).parent / 'data' / 'ideal'


TEST_CASES = {
    'boxcar': {
        'default': {'points': np.array([[1.], [2.]]),
                    'mesh': np.linspace(-5, 5, 11),
                    'kwargs': {'width': 3.}},
        'peak': {'points': np.array([[0.], [2.]])},
        'broaden': {'mesh': np.linspace(0, 5, 11),  # i.e. each bin is 0.5
                    'points': np.linspace(0, 5, 11)[:, None],
                    'data': np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=float),
                    'kwargs': {'width': 1.}},  # i.e. pulse area = 1}
    },
    'triangle': {
        'default': {
            'points': np.array([[3.], [7.]]),
            'mesh': np.arange(0., 10., 1.),
            'data': np.array([0.5, 2.]),
            'kwargs': {'fwhm': 2.}
        },
        'kernel': {
            'mesh': np.arange(-5, 5, 1.)
        }
    },
    'trapezoid': {
        'default': {
            'points': np.array([[3.], [7.]]),
            'mesh': np.arange(-2., 10., 1.),
            'data': np.array([0.5, 2.]),
            'kwargs': {'long_base': 6., 'short_base': 2.},
        },
        'kernel': {'mesh': np.linspace(-6, 6, 13)}
    },
    'gaussian': {
        'default': {
            'points': np.array([[3.], [7.]]),
            'mesh': np.linspace(0., 10., 41),
            'data': np.array([0.5, 2.]),
            'kwargs': {'sigma': 2.}
        },
        'kernel': {'mesh': np.linspace(-6, 6, 13)}
    },
    'lorentzian': {
        'default': {
            'points': np.array([[3.], [7.]]),
            'mesh': np.linspace(0., 10., 41),
            'data': np.array([0.5, 2.]),
            'kwargs': {'fwhm': 2.}
        },
        'kernel': {'mesh': np.linspace(-6, 6, 13)}
    }
}


class Feature(StrEnum):
    KERNEL = 'kernel'
    PEAK = 'peak'
    BROADEN = 'broaden'


test_specs = list(
    product(("boxcar", "triangle", "trapezoid", "gaussian", "lorentzian"), Feature)
)


def _get_data(name: str, feature: Feature):
    params = ChainMap(TEST_CASES[name].get(feature, {}),
                      TEST_CASES[name].get('default', {}))
    instrument = Instrument.from_default('IDEAL')
    model = instrument.get_resolution_function(name, **params.get('kwargs', {}))

    match feature:
        case Feature.KERNEL:
            result = model.get_kernel(params['points'], params['mesh'])
        case Feature.PEAK:
            result = model.get_peak(params['points'], params['mesh'])
        case Feature.BROADEN:
            result = model.broaden(params['points'], params['data'], params['mesh'])
        case _:
            raise ValueError()

    return result, params['mesh']


@pytest.mark.parametrize("name,feature", test_specs)
def test_ideal_model(name: str, feature: Feature):
    result, _ = _get_data(name, feature)
    assert_allclose(result, np.load(DATA_PATH / f'_get_{name}_{feature}.npy'))

def generate_data():
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt

    DATA_PATH.mkdir(exist_ok=True)

    for name, feature in test_specs:
        print(f'Generating test data: {name}, {feature}')

        result, mesh = _get_data(name, feature)
        np.save(DATA_PATH / f'_get_{name}_{feature}.npy', result)

        fig, ax = plt.subplots()

        fmt = '-o' if len(mesh) < 20 else '-'

        if len(np.shape(result)) == 1:
            ax.plot(mesh, result, fmt)
        else:
            for row in result:
                ax.plot(mesh, row, fmt)

        ax.set_title(f"{name}: {feature}")
        fig.tight_layout()
        fig.savefig(DATA_PATH / f'_get_{name}_{feature}.png')
        plt.close(fig)


if __name__ == '__main__':
    generate_data()
