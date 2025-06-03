import argparse
import glob
from itertools import product
import logging
import os
import warnings

import yaml

from resolution_functions import Instrument


class ValidationError(Exception):
    pass


def validate_top_level(data: dict, header: str) -> None:
    """
    Validates the top-level of the `data` dictionary.

    Parameters
    ----------
    data
        The dictionary containing the all the data from a YAML data file.
    header
        The header to prepend to error messages - should include the file name.
    """
    if not 'name' in data:
        raise ValidationError(f'{header}: the "name" key is missing.')
    elif not isinstance(data['name'], str):
        raise ValidationError(f'{header}: the "name" must be a str (is a {type(data["name"])})')

    if not 'version' in data:
        raise ValidationError(f'{header}: the "version" key is missing.')
    elif not isinstance(data['version'], dict):
        raise ValidationError(f'{header}: the "version" must be a (YAML) dict (is a '
                              f'{type(data["version"])})')

    if not 'default_version' in data:
        raise ValidationError(f'{header}: the "default_version" key is missing.')
    elif not isinstance(data['default_version'], str):
        raise ValidationError(f'{header}: the "default_version" must be a str (is a '
                              + str(type(data['default_version'])) + ')')
    elif not data['default_version'] in data['version']:
        raise ValidationError(f'{header}: the "default_version" must correspond to one of the '
                              f'specified versions. I.e. "{data["default_version"]}" not in '
                              f'{list(data["version"].keys())}')

    validate_version_dict(data['version'], header)


def validate_version_dict(version_dict: dict, header: str) -> None:
    """
    Validates the second level of the data file, the dictionary inside the ``version`` key.

    Parameters
    ----------
    version_dict
        The dictionary containing the data inside the ``version`` key of the YAML data file.
    header
        The header to prepend to error messages.
    """
    if not version_dict:
        raise ValidationError(f'{header}: No versions provided (dictionary is empty)')

    for version in version_dict.values():
        header = f'{header}: version "{version}": '
        if not isinstance(version, dict):
            raise ValidationError(f'{header}must be a (YAML) dict (is a {type(version)})')

        validate_version(version, header)


def validate_version(version: dict, header: str) -> None:
    """
    Validates the third level of the data file, the dictionary inside a key representing a
    particular version.

    Parameters
    ----------
    version
        The dictionary containing the data inside a key representing a particular version in the
        YAML data file.
    header
        The header to prepend to error messages.
    """
    if not 'models' in version:
        raise ValidationError(f'{header}the "models" key is missing.')
    elif not isinstance(version['models'], dict):
        raise ValidationError(f'{header}the "models" must be a str (is a {type(version["name"])})')

    validate_models_dict(version['models'], header)


def validate_models_dict(models_dict: dict, header: str) -> None:
    """
    Validates the fourth level of the data file, the dictionary inside the ``models`` key.

    Parameters
    ----------
    models_dict
        The dictionary containing the data inside the ``models`` key of the YAML data file.
    header
        The header to prepend to error messages.
    """
    if not models_dict:
        raise ValidationError(f'{header}No models provided (dictionary is empty)')

    for model_name, model_value in models_dict.items():
        header = f'{header}: model "{model_name}": '
        split = model_name.split('_')

        if isinstance(model_value, str):
            if len(split) > 1:
                try:
                    result = int(split[-1][1:])
                except ValueError:
                    result = None

                if result is not None:
                    if split[-1][0] == 'v':
                        raise ValidationError(f'{header} the name of a model that points to a '
                                              f'concrete (versioned) model must not have a version '
                                              f'in its name, i.e. be without e.g. "_v1".')
                    else:
                        warnings.warn(f'{header} the name of a model that points to a concrete '
                                      f'(versioned) model should not normally end with an _integer',
                                      Warning)

            if not model_value in models_dict:
                raise ValidationError(f'{header} does not point to a specified model. I.e. '
                                      f'"{model_value}" not in {list(models_dict.keys())}')
            elif not isinstance(models_dict[model_value], dict):
                raise ValidationError(f'{header} must point to a versioned, fully specified model. '
                                      f'I.e. "{model_value}" is not a (YAML) dict.')

        elif isinstance(model_value, dict):
            if len(split) < 2:
                raise ValidationError(f'{header} a versioned model must end in a version number '
                                      f'(e.g. "_v1")')

            if split[-1][0] != 'v':
                raise ValidationError(f'{header}: "v" missing - a versioned model must end in a '
                                      f'version number (e.g. "_v1")')

            try:
                int(split[-1][1:])
            except ValueError:
                raise ValidationError(f'{header} does not end in an integer - a versioned model '
                                      f'must end in a version number (e.g. "_v1") where the number '
                                      f'is an integer.')

            validate_model(model_value, header)
        else:
            raise ValidationError(f'{header}a model must be either a str or a (YAML) dict (is a '
                                  f'{type(model_value)})')


def validate_model(model: dict, header: str) -> None:
    """
    Validates the fifth level of the data file, the dictionary inside a key representing a
    particular model.

    Parameters
    ----------
    model
        The dictionary containing the data inside a key representing a particular model inside the
        YAML data file.
    header
        The header to prepend to error messages.
    """
    if not 'function' in model:
        raise ValidationError(f'{header}the "function" key is missing.')
    elif not isinstance(model['function'], str):
        raise ValidationError(f'{header}the "function" must be a str (is a '
                              f'{type(model["function"])})')

    if not 'citation' in model:
        raise ValidationError(f'{header}the "citation" key is missing.')
    elif not isinstance(model['citation'], list):
        raise ValidationError(f'{header}the "citation" must be a list of str (is a '
                              f'{type(model["citation"])})')
    else:
        for value in model['citation']:
            if not isinstance(value, str):
                raise ValidationError(f'{header}the "citation" must be a list of str (entry '
                                      f'"{value}" is a {type(model["citation"])})')

    if not 'parameters' in model:
        raise ValidationError(f'{header}the "parameters" key is missing.')
    elif not isinstance(model['parameters'], dict):
        raise ValidationError(f'{header}the "parameters" must be a (YAML) dict (is a '
                              f'{type(model["parameters"])})')

    if not 'configurations' in model:
        raise ValidationError(f'{header}the "configurations" key is missing.')
    elif not isinstance(model['configurations'], dict):
        raise ValidationError(f'{header}the "configurations" must be a (YAML) dict (is a '
                              f'{type(model["configurations"])})')

    validate_configurations_dict(model['configurations'], header)


def validate_configurations_dict(configurations_dict: dict, header: str) -> None:
    """
    Validates the sixth level of the data file, the dictionary inside the ``configurations`` key.

    Parameters
    ----------
    configurations_dict
        The dictionary containing the data inside the ``configurations`` key of the YAML data file.
    header
        The header to prepend to the error messages.
    """
    if not configurations_dict:
        # No configurations, nothing to validate. This is allowed.
        return

    for name, config in configurations_dict.items():
        header = f'{header}: configuration "{config}": '

        if not isinstance(config, dict):
            raise ValidationError(f'{header}must be a (YAML) dict (is a {type(config)})')

        validate_configuration(config, header)


def validate_configuration(configuration: dict, header: str) -> None:
    """
    Validates the seventh level of the data file, the dictionary inside a key representing a
    particular configuration.

    Parameters
    ----------
    configuration
        The dictionary containing the data inside a key corresponding to a particular configuration
        inside the YAML data file.
    header
        The header to prepend to the error messages.
    """
    if not 'default_option' in configuration:
        raise ValidationError(f'{header}the "default_option" key is missing.')
    elif not isinstance(configuration['default_option'], str):
        raise ValidationError(f'{header}the "default_option" must be a str (is a '
                              f'{type(configuration["default_option"])})')
    elif not configuration['default_option'] in configuration:
        raise ValidationError(f'{header}the "default_option" must correspond to one of the provided'
                              f' options. I.e. "{configuration["default_option"]}" not in '
                              f'{list(configuration.keys())}')

    if len(configuration.keys()) < 2:
        raise ValidationError(f'{header}must have at least one option.')

    for name, option in configuration.items():
        if name == 'default_option':
            continue

        if not isinstance(option, dict):
            raise ValidationError(f'{header}Option "{option}" must be a dict (is a {type(option)})')


def validate_with_resins(data: dict) -> None:
    """
    Validates the ``data`` dictionary using ResINS by attempting to create the instrument and
    ``ModelData``.

    Parameters
    ----------
    data
        The dictionary from a YAML data file containing all the data.
    """
    all_versions = data['version']
    name = data['name']

    for version, version_data in all_versions.items():
        instrument = Instrument(
            name,
            version,
            version_data['models'],
            version_data['default_model']
        )

        for model in instrument.available_models:
            configs = instrument.possible_configurations_for_model(model)
            options = [instrument.possible_options_for_model_and_configuration(model, config)
                       for config in configs]

            for combination in product(*options):
                kwargs = {config: option for config, option in zip(configs, combination)}
                try:
                    instrument.get_model_data(model_name=model, **kwargs)
                except Exception as e:
                    raise ValidationError(
                        f'Could not construct ModelData for instrument "{name}", version '
                        f'"{version}", model "{model}", and options "{kwargs}" because of '
                    ) from e

                try:
                    instrument.get_resolution_function(model_name=model, **kwargs)
                except Exception as e:
                    raise ValidationError(
                        f'Could not run get_resolution_function for instrument "{name}", version '
                        f'"{version}", model "{model}", and options "{kwargs}" (and the remaining '
                        f'values DEFAULT) because of '
                    ) from e


def main(path: str,
         logger: logging.Logger,
         skip_yaml: bool = False,
         skip_resins: bool = False) -> None:
    """
    Validates (all) the file(s) found at `path`.

    Parameters
    ----------
    path
        The path to the file(s) to validate.
    logger
        The logger to use for logging
    skip_yaml
        If True, disables YAML data validation by dedicated validator
    skip_resins
        If True, disables file validation by ResINS
    """
    files = glob.glob(path)

    for file in files:
        logger.info(file)

        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        if not skip_yaml:
            validate_top_level(data, os.path.split(file)[-1])

        if not skip_resins:
            validate_with_resins(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    default = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(default, 'src', 'resolution_functions', 'instrument_data', '*.yaml')

    parser.add_argument('path', nargs='?', default=default,
                        help='The path to the YAML data file(s) to validate. May include globs.')
    parser.add_argument('-dy', '--disable-yaml', action='store_true',
                        help='Disables YAML data validation by dedicated validator')
    parser.add_argument('-dr', '--disable-resins', action='store_true',
                        help='Disables file validation by ResINS')
    parser.add_argument('-l', '--log-level', default='INFO', type=str)

    args = parser.parse_args()

    logger = logging.getLogger()
    logging.basicConfig(level=args.log_level)

    main(args.path, logger, args.disable_yaml, args.disable_resins)
