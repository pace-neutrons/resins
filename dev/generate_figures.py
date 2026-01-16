# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "resins",
# ]
#
# [tool.uv.sources]
# resins = { path = "../" }
# ///

"""Generate schematic figures used in docs"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from resins import Instrument


figures = Path(__file__).parent.parent / 'docs/source/figures'

def boxcar_demo() -> None:
    width = 2.3

    x_range = (-3, 3.01)
    spacings = [0.1, 0.4, 1.]

    fig, axes = plt.subplots(nrows=len(spacings), sharex=True, figsize=(4, 4))

    for spacing, ax in zip(spacings, axes):
        x = np.arange(*x_range, spacing)
        y = Instrument.from_default(
            "IDEAL"
        ).get_resolution_function(
            "boxcar", width=width
        ).get_kernel(np.array([[0.]]), mesh=x)[0]

        ax.hlines(1/width, *x_range, linestyle='--')

        ax.set_xlim(*x_range)
        ax.set_ylim(0, 0.5)

        ax.plot(x, y, '-o')

    fig.savefig(figures / 'boxcar_binwidth.png')


def triangle_demo() -> None:

    fig, axes = plt.subplots(nrows=2, figsize=(6, 4))

    kernel_widths = 1, 1.2, 3
    kernel_mesh = np.linspace(-4, 4, 9)

    for fwhm in kernel_widths:
        y = Instrument.from_default(
            "IDEAL",
        ).get_resolution_function(
            "triangle", fwhm=fwhm,
        ).get_kernel(np.array([[0.]]), mesh=kernel_mesh)[0]

        axes[0].plot(kernel_mesh, y, 'o-', label=f'FWHM {fwhm}')

    axes[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    axes[0].set_title('get_kernel()')

    peak_mesh = np.arange(0, 12, 1)
    peak_positions = [2, 5.4, 8.6]
    fwhm = 2

    y = Instrument.from_default(
            "IDEAL",
        ).get_resolution_function(
            "triangle", fwhm=fwhm,
        ).get_peak(np.array(peak_positions)[:, None], mesh=peak_mesh)

    for position, peak, color in zip(peak_positions, y, ['C0', 'C1', 'C2']):
        axes[1].plot(peak_mesh, peak, 'o-', color=color, label=f'Position: {position}')
        axes[1].axvline(position, color=color, linestyle='--')

    axes[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    axes[1].set_title('get_peak()')

    fig.tight_layout()
    fig.savefig(figures / 'triangle_peaks.png')


def main() -> None:
    boxcar_demo()
    
    triangle_demo()

if __name__ == "__main__":
    main()
