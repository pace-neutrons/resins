# Resolution Functions

Python library for working with resolution functions of inelastic neutron scattering (INS) 
instruments. This package exists to centralise all things related to resolution of INS instruments 
and make it easier to work with. It pools related code from existing projects, namely 
[AbINS](https://github.com/mantidproject/mantid/tree/main/scripts/abins) and 
[PyChop](https://github.com/mducle/pychop/tree/main), as well as implementing code published in 
literature. The main purposes are:

1. Provide one, central place implementing various models for INS instruments (resolution functions)
2. Provide a simple way to obtain the broadening at a given frequency, for a given instrument and settings
3. Provide a way to apply broadening to a spectrum

## Quick Start

The package can be installed with pip (see Installation). To start, import the main `Instrument` 
class and get the instrument object of your choice:

```
>>> from resolution_functions import Instrument
>>> maps = Instrument.from_default('MAPS', 'MAPS')
>>> print(maps)
Instrument(name=MAPS, version=MAPS)
```

To get the resolution function, call the `get_resolution_function` method (providing all your 
choices for the required settings and configurations), which returns a callable that can be called 
to broaden the data.

```
>>> # The available models for a given instrument can be queried:
>>> maps.available_models
['PyChop_fit']
>>> # There are multiple ways of querying the model-specific parameters, but the most comprehensive is
>>> maps.get_model_signature('PyChop_fit')
<Signature (model_name: Optional[str] = 'PyChop_fit_v1', *, chopper_package: Literal['A', 'B', 'S'] = 'A', e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500, chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400, fitting_order: 'int' = 4, _) -> resolution_functions.models.pychop.PyChopModelFermi>
>>> # Now we can get the resolution function
>>> pychop = maps.get_resolution_function('PyChop_fit', chopper_package='B', e_init=500, chopper_frequency=300)
>>> print(pychop)
PyChopModelFermi(citation=[''])
```

Calling the model (like a function) broadens the data at the provided combinations of energy 
transfer and momentum ([w, Q]), using a mesh and the corresponding data:

```
>>> import numpy as np
>>> energy_transfer = np.array([100, 200, 300])[:, np.newaxis]
>>> data = np.array([0.6, 1.5, 0.9])
>>> mesh = np.linspace(0, 500, 1000)
>>> pychop(energy_transfer, data, mesh)
array([3.43947518e-028, ... 5.99877942e-002, ... 7.31766110e-249])
```

However, the model also provides methods that go lower; 

- `get_kernel` computes the broadening kernel at each [w, Q] (centered on 0)
- `get_peak` computes the broadening peak at each [w, Q] (centered on the [w, Q])
- `get_characteristics` returns only the characteristic parameters of the kernel 
  at each [w, Q] (such as the standard deviation of the normal distribution)

```
>>> peaks = pychop.get_peak(energy_transfer, mesh)
>>> mesh_centered_on_0 = np.linspace(-100, 100, 1000)
>>> kernels = pychop.get_kernel(energy_transfer, mesh_centered_on_0)
>>> pychop.get_characteristics(energy_transfer)
{'sigma': array([9.15987016, 7.38868127, 5.93104319])}
```

## Installation

This package can be installed using pip, though it is not yet on PyPI, so it has to be installed directly from GitHub:

```
pip install git+https://github.com/pace-neutrons/resolution_functions.git
```

or from a local copy:

```
git clone https://github.com/pace-neutrons/resolution_functions.git
pip install resolution_functions
```


