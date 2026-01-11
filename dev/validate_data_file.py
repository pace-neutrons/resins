import argparse
import logging
from pathlib import Path
import warnings
from itertools import product
from collections.abc import Iterable

import yaml

from schema import And, Or, Regex, Schema, Hook, SchemaError, Optional
from resolution_functions import Instrument


VersionVWarning = (
    "{key} the name of a model that points to a concrete "
    "(versioned) model should not normally end with an _integer"
)


class ValInParent(Hook):
    """Check whether a referenced key exists.

    Parameters
    ----------
    loc : str, optional
        Subkey to search under. If ``None`` search same level as current.
    """
    def __init__(self, *args, loc: str | None = None, **kwargs) -> None:
        self._loc = loc
        super().__init__(*args, handler=self._val_in_container, **kwargs)

    def _val_in_container(self, key: str, scope: dict, error: str) -> None:
        ref = scope[key]
        check = scope[self._loc] if self._loc is not None else scope
        if ref not in check:
            raise KeyError(f"{ref} not in containing scope.")


class WarnOnRead(Hook):
    """Issue a warning if key present.

    Parameters
    ----------
    warning : str
        Warning message to issue.
    warning_level : Warning
        Class of warning to use.
    """
    def __init__(self, *args, warning: str, warning_level: Warning = Warning, **kwargs) -> None:
        self._warning = warning
        self._warning_level = warning_level
        super().__init__(*args, handler=self._warn, **kwargs)

    def _warn(self, key: str, scope: dict, error: str) -> None:
        warnings.warn(self._warning.format(key=key, scope=scope, error=error), self._warning_level)


MODEL = Schema(
    {
        "function": And(str, len),
        "citation": [str],
        "parameters": dict,
        "configurations": {
            str: {
                ValInParent("default_option"): str,
                "default_option": str,
                str: dict,
            }
        },
    },
)

VersionSchema = Or(
    Regex(r"_v\d+$"), WarnOnRead(Regex(r"_\d+$"), warning=VersionVWarning), Regex(r"_\d+$"),
)
VERSION = Schema(
    {
        str: {
            "models": {
                ValInParent(str): str,  # If it's a string, it must be in a parent.
                Optional(VersionSchema): Or(MODEL, str),
                Optional(str): str,
            },
        },
    },
    ignore_extra_keys=True,
)


TOP_LEVEL = Schema(
    {
        "name": And(str, len),
        "version": VERSION,
        ValInParent("default_version", loc="version"): str,
        "default_version": str,
    },
)


def validate(data: dict) -> None:
    """
    Validates the `data` dictionary.

    Parameters
    ----------
    data
        The dictionary containing the all the data from a YAML data file.
    """
    TOP_LEVEL.validate(data)


def validate_with_resins(data: dict) -> None:
    """
    Validates the ``data`` dictionary using ResINS by attempting to create the instrument and
    ``ModelData``.

    Parameters
    ----------
    data
        The dictionary from a YAML data file containing all the data.
    """
    all_versions = data["version"]
    name = data["name"]

    for version, version_data in all_versions.items():
        instrument = Instrument(
            name, version, version_data["models"], version_data["default_model"]
        )

        for model in instrument.available_models:
            configs = instrument.possible_configurations_for_model(model)
            options = [
                instrument.possible_options_for_model_and_configuration(model, config)
                for config in configs
            ]

            for combination in product(*options):
                kwargs = {
                    config: option for config, option in zip(configs, combination)
                }
                try:
                    instrument.get_model_data(model_name=model, **kwargs)
                except Exception as e:
                    raise SchemaError(
                        f'Could not construct ModelData for instrument "{name}", version '
                        f'"{version}", model "{model}", and options "{kwargs}" because of '
                    ) from e

                try:
                    instrument.get_resolution_function(model_name=model, **kwargs)
                except Exception as e:
                    raise SchemaError(
                        f'Could not run get_resolution_function for instrument "{name}", version '
                        f'"{version}", model "{model}", and options "{kwargs}" (and the remaining '
                        f"values DEFAULT) because of "
                    ) from e


def _default_paths():
    """Get instrument data yaml files from this package"""
    data_dir = Path(__file__).parent.parent / "src/resolution_functions/instrument_data"
    return list(data_dir.glob("*.yaml"))


def main(
    paths: Iterable[Path],
    logger: logging.Logger,
    *,
    skip_yaml: bool = False,
    skip_resins: bool = False,
) -> None:
    """
    Validates (all) the file(s) found at `path`.

    Parameters
    ----------
    paths
        Paths of file(s) to validate.
    logger
        The logger to use for logging
    skip_yaml
        If True, disables YAML data validation by dedicated validator
    skip_resins
        If True, disables file validation by ResINS
    """
    for path in paths:
        logger.info(path)

        with open(path, "r") as fd:
            data = yaml.safe_load(fd)

        if not skip_yaml:
            validate(data)

        if not skip_resins:
            validate_with_resins(data)


def cli() -> None:
    """Run validation through command-line interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "paths",
        nargs=argparse.REMAINDER,
        type=Path,
        help="YAML data file(s) to validate.",
    )
    parser.add_argument(
        "-dy",
        "--disable-yaml",
        action="store_true",
        help="Disables YAML data validation by dedicated validator",
    )
    parser.add_argument(
        "-dr",
        "--disable-resins",
        action="store_true",
        help="Disables file validation by ResINS",
    )
    parser.add_argument("-l", "--log-level", default="INFO", type=str)

    args = parser.parse_args()

    if not args.paths:
        setattr(args, "paths", _default_paths())

    logger = logging.getLogger()
    logging.basicConfig(level=args.log_level)

    main(
        args.paths, logger, skip_yaml=args.disable_yaml, skip_resins=args.disable_resins
    )


if __name__ == "__main__":
    cli()
