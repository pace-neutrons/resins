import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from PyChop.Instruments import Instrument as PyChopInstrument

import mantid
from abins.instruments.pychop import PyChopInstrument as AbINSPyChopInstrument

from resins.instrument import Instrument
from resins.models.pychop import SIGMA2FWHM

WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV

instrument = ('MAPS', 'MAPS')
setting = 'S'

energy = 50
frequency = 100



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots a comparison between this implementation and AbINS and PyChop.')
    parser.add_argument('-i', '--instrument', type=str, default=instrument[0],
                        help=f'The instrument to compare. Only the PyChop instruments are possible. '
                             f'The default is {instrument[0]}')
    parser.add_argument('-v', '--version', type=str, default=instrument[1],
                        help=f'The version of the instrument, as defined in this implementation. '
                             f'The default is {instrument[1]}')
    parser.add_argument('-c', '--chopper', type=str, default=setting,
                        help=f'The chopper package to use. The default is {setting}')
    parser.add_argument('-e', '--energy', type=float, default=energy,
                        help=f'The incident energy to use. The default is {energy}')
    parser.add_argument('-f', '--frequency', type=float, default=frequency,
                        help=f'The chopper frequency to use. The default is {frequency}')
    parser.add_argument('--evaluation', type=int, default=1000,
                        help='The number of points to use for evaluation and plotting. The default is 1000')
    parser.add_argument('-o', '--output', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pychop.png'),
                        help='The path to which to save the plotted image. Default is "pychop.png" in the directory of '
                             'this script.')

    args = parser.parse_args()

    energies = np.linspace(0, energy, args.evaluation)
    instrument = (args.instrument, args.version)

    abins = AbINSPyChopInstrument(name=instrument[0], setting=args.chopper, chopper_frequency=args.frequency)
    abins.set_incident_energy(args.energy, 'meV')
    abins = abins.calculate_sigma(energies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    rf = Instrument.from_default(*instrument)
    rf = rf.get_resolution_function('PyChop_fit', chopper_package=args.chopper, e_init=args.energy, chopper_frequency=args.frequency)
    rf = rf(energies)

    pychop = PyChopInstrument(instrument[0], chopper=args.chopper, freq=args.frequency)
    pychop = pychop.getResolution(Ei_in=args.energy, Etrans=energies.tolist())
    pychop /= SIGMA2FWHM

    fig, ax = plt.subplots(dpi=500)

    ax.plot(energies, pychop, c='black', label='pychop')
    ax.plot(energies, abins, c='blue', alpha=0.5, label='abins')
    ax.plot(energies, rf, c='red', alpha=0.5, label='resolution functions')

    plt.legend()

    plt.savefig(args.output)
