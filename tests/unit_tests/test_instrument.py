from dataclasses import dataclass
import inspect
import os
import typing

import pytest
import yaml

from resolution_functions import instrument as i
from resolution_functions.models import MODELS
from resolution_functions.models.model_base import ModelData, InstrumentModel


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_YAML = os.path.join(TEST_DIR, 'fake_instrument.yaml')


@dataclass(init=True, repr=True, frozen=True, slots=True)
class MockModelData(ModelData):
    param1: int
    param_2: float
    param3: float
    param4: int
    param5: int
    param6: float
    string: str
    lst: list[float | int]
    matrix: list[list[int]]
    dictionary: dict[str, dict[str, int] | int]


class MockModel(InstrumentModel):
    input = 1
    output = 1
    data_class = MockModelData

    def __init__(self,
                 model_data: MockModelData,
                 arg1: int,
                 kwarg1: None = None,
                 kwarg2: int = 0, **_):
        super().__init__(model_data)
        self.data = model_data
        self.arg1 = arg1
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2

    def get_characteristics(self, energy_transfer):
        return {}

    def __call__(self, frequencies, *args, **kwargs):
        return frequencies


@pytest.fixture
def mock_models():
    return {'mock': MockModel}

@pytest.fixture
def mock_instrument_map():
    return {
        'TEST': ('fake_instrument.yaml', None),
        'ALIAS': ('fake_instrument.yaml', 'VERSION1'),
    }


@pytest.fixture(scope='module')
def data():
    with open(FAKE_YAML, 'r') as f:
        data = yaml.safe_load(f)

    return data


@pytest.fixture(scope='module')
def instrument(data):
    return i.Instrument(
        data['name'],
        'VERSION1',
        data['version']['VERSION1']['models'],
        data['version']['VERSION1']['default_model']
    )


@pytest.fixture
def version1_mock_v3_data_default(data):
    model = data['version']['VERSION1']['models']['mock_v3']
    return MockModelData(
        function=model['function'],
        citation=model['citation'],
        **model['parameters'],
        **model['configurations']['config1']['A'],
        **model['configurations']['config2']['X'],
    )


@pytest.fixture
def version1_mock_v3_data_nondefault(data):
    model = data['version']['VERSION1']['models']['mock_v3']
    return MockModelData(
        function=model['function'],
        citation=model['citation'],
        **model['parameters'],
        **model['configurations']['config1']['C'],
        **model['configurations']['config2']['Y'],
    )

@pytest.fixture(scope='module')
def test_instrument(data):
    return i.Instrument(
        data['name'],
        'TEST',
        data['version']['TEST']['models'],
        data['version']['TEST']['default_model']
    )


def test_available_instruments(mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    assert i.Instrument.available_instruments() == ['TEST', 'ALIAS']


def test_private_available_versions(mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    actual_versions, actual_default = i.Instrument._available_versions(FAKE_YAML)

    assert actual_default == 'TEST'
    assert actual_versions == ['VERSION1', 'TEST']


def test_available_versions(mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    actual_versions, actual_default = i.Instrument.available_versions('TEST')

    assert actual_default == 'TEST'
    assert actual_versions == ['VERSION1', 'TEST']


def test_available_versions_alias(mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    actual_versions, actual_default = i.Instrument.available_versions('ALIAS')

    assert actual_default == 'VERSION1'
    assert actual_versions == ['VERSION1', 'TEST']


def test_from_file(data):
    instrument = i.Instrument.from_file(FAKE_YAML, 'VERSION1')

    assert isinstance(instrument, i.Instrument)
    assert instrument.name == 'TEST'
    assert instrument.version == 'VERSION1'
    assert instrument._models == data['version']['VERSION1']['models']


def test_from_file_default_version(data):
    instrument = i.Instrument.from_file(FAKE_YAML)

    assert isinstance(instrument, i.Instrument)
    assert instrument.name == 'TEST'
    assert instrument.version == 'TEST'
    assert instrument._models == data['version']['TEST']['models']


def test_from_file_invalid_version():
    with pytest.raises(i.InvalidVersionError):
        i.Instrument.from_file(FAKE_YAML, 'INVALID_VERSION')


def test_from_default(data, mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    instrument = i.Instrument.from_default('TEST', 'VERSION1')

    assert isinstance(instrument, i.Instrument)
    assert instrument.name == 'TEST'
    assert instrument.version == 'VERSION1'
    assert instrument._models == data['version']['VERSION1']['models']


def test_from_default_default(data, mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    instrument = i.Instrument.from_default('TEST')

    assert isinstance(instrument, i.Instrument)
    assert instrument.name == 'TEST'
    assert instrument.version == 'TEST'
    assert instrument._models == data['version']['TEST']['models']


@pytest.mark.parametrize("name,expected_path,implied_ver",
                         [('TEST', FAKE_YAML, None), ('ALIAS', FAKE_YAML, 'VERSION1')])
def test_get_file(name, expected_path, implied_ver, mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    actual_path, actual_version = i.Instrument._get_file(name)

    assert actual_path == expected_path
    assert actual_version == implied_ver


def test_get_file_invalid(mock_instrument_map, mocker):
    mocker.patch('resolution_functions.instrument.INSTRUMENT_MAP', mock_instrument_map)
    mocker.patch('resolution_functions.instrument.INSTRUMENT_DATA_PATH', TEST_DIR)

    with pytest.raises(i.InvalidInstrumentError):
        i.Instrument._get_file('INVALID_INSTRUMENT')


def test_private_get_model_data(instrument, version1_mock_v3_data_nondefault, mock_models, mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    actual_model, actual_name = instrument._get_model_data(model_name='mock_v3', config1='C', config2='Y')

    assert actual_name == 'mock_v3'
    assert isinstance(actual_model, MockModelData)
    assert actual_model == version1_mock_v3_data_nondefault


def test_private_get_model_data_default(instrument,
                                        version1_mock_v3_data_default,
                                        mock_models,
                                        mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    actual_model, actual_name = instrument._get_model_data()

    assert actual_name == 'mock_v3'
    assert isinstance(actual_model, MockModelData)
    assert actual_model == version1_mock_v3_data_default


def test_private_get_model_data_invalid_model(test_instrument):
    with pytest.raises(i.InvalidModelError):
        test_instrument._get_model_data(model_name='invalid_model')


def test_get_model_data(instrument, version1_mock_v3_data_nondefault, mock_models, mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    actual_model = instrument.get_model_data(model_name='mock_v3', config1='C', config2='Y')

    assert isinstance(actual_model, MockModelData)
    assert actual_model == version1_mock_v3_data_nondefault


def test_get_model_data_invalid_type_no_error(instrument,
                                              version1_mock_v3_data_nondefault,
                                              mock_models,
                                              mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    # mock_v1 has param1 with invalid type
    actual_model = instrument.get_model_data(model_name='mock_v1', config1='C', config2='Y')

    assert isinstance(actual_model, MockModelData)


def test_resolve_model(instrument, data):
    actual_model, actual_name = instrument._resolve_model('mock')

    assert actual_model == data['version']['VERSION1']['models']['mock_v3']
    assert actual_name == 'mock_v3'


def test_resolve_model_invalid_model(instrument):
    with pytest.raises(i.InvalidModelError):
        instrument._resolve_model('INVALID_MODEL')


def test_resolve_model_invalid_alias(test_instrument):
    with pytest.raises(i.InvalidModelError):
        test_instrument._resolve_model('mock')


def test_get_resolution_function(instrument, version1_mock_v3_data_nondefault, mock_models, mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    arg1, kwarg2 = True, 42
    actual = instrument.get_resolution_function(
        'mock_v3', config1='C', config2='Y', arg1=arg1, kwarg2=kwarg2, unused=False
    )

    assert isinstance(actual, MockModel)
    assert actual.data == version1_mock_v3_data_nondefault
    assert actual.arg1 == arg1
    assert actual.kwarg1 is None
    assert actual.kwarg2 == kwarg2


def test_get_resolution_function_invalid_function(instrument, mock_models, mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    with pytest.raises(KeyError):
        instrument.get_resolution_function('mock_v2')


def test_get_model_signature(instrument, mock_models, mocker):
    mocker.patch('resolution_functions.instrument.MODELS', mock_models)

    expected_names = ['model_name', 'config1', 'config2', 'arg1', 'kwarg1', 'kwarg2']
    expected_annotations = [typing.Optional[str], typing.Literal['A', 'B', 'C'],
                            typing.Literal['X', 'Y'], int, None,
                            typing.Annotated[int, 'restriction=[0]']]
    expected_defaults = ['mock_v3', 'A', 'X', inspect.Signature.empty, None, 0]
    actual = instrument.get_model_signature('mock_v3')

    assert isinstance(actual, inspect.Signature)
    assert actual.return_annotation == MockModel

    for exp_name, exp_ann, exp_def in zip(expected_names, expected_annotations, expected_defaults):
        assert actual.parameters[exp_name].name == exp_name
        assert actual.parameters[exp_name].annotation == exp_ann
        assert actual.parameters[exp_name].default == exp_def


def test_available_models(instrument):
    assert instrument.available_models == ['mock', 'empty']



def test_all_available_models(instrument):
    expected = ['mock', 'mock_v1', 'mock_v2', 'mock_v3', 'empty', 'empty_v1']
    assert instrument.all_available_models == expected


def test_available_unique_models(instrument):
    assert instrument.available_unique_models == ['mock_v1', 'mock_v2', 'mock_v3', 'empty_v1']


def test_format_available_models_and_configurations(instrument):
    expected = """\
|-------------|--------------|-------------------|
| MODEL       | ALIAS FOR    | CONFIGURATIONS    |
|=============|==============|===================|
| mock        | mock_v3      |                   |
|-------------|--------------|-------------------|
| mock_v1     |              | config1           |
|             |              | config2           |
|-------------|--------------|-------------------|
| mock_v2     |              | config1           |
|             |              | config2           |
|-------------|--------------|-------------------|
| mock_v3     |              | config1           |
|             |              | config2           |
|-------------|--------------|-------------------|
| empty       | empty_v1     |                   |
|-------------|--------------|-------------------|
| empty_v1    |              |                   |
|-------------|--------------|-------------------|
"""

    assert instrument.format_available_models_and_configurations() == expected


def test_format_available_models_options(instrument):
    expected = """\
|-------------|--------------|-------------------|----------------|
| MODEL       | ALIAS FOR    | CONFIGURATIONS    | OPTIONS        |
|=============|==============|===================|================|
| mock        | mock_v3      |                   |                |
|-------------|--------------|-------------------|----------------|
| mock_v1     |              | config1           | A (default)    |
|             |              |                   | B              |
|             |              |                   | C              |
|             |              |                   |                |
|             |              | config2           | X (default)    |
|             |              |                   | Y              |
|-------------|--------------|-------------------|----------------|
| mock_v2     |              | config1           | A (default)    |
|             |              |                   | B              |
|             |              |                   | C              |
|             |              |                   |                |
|             |              | config2           | X (default)    |
|             |              |                   | Y              |
|-------------|--------------|-------------------|----------------|
| mock_v3     |              | config1           | A (default)    |
|             |              |                   | B              |
|             |              |                   | C              |
|             |              |                   |                |
|             |              | config2           | X (default)    |
|             |              |                   | Y              |
|-------------|--------------|-------------------|----------------|
| empty       | empty_v1     |                   |                |
|-------------|--------------|-------------------|----------------|
| empty_v1    |              |                   |                |
|-------------|--------------|-------------------|----------------|\n"""

    assert instrument.format_available_models_options() == expected


def test_possible_configurations_for_model(instrument):
    assert instrument.possible_configurations_for_model('mock_v3') == ['config1', 'config2']


def test_possible_options_for_model(instrument):
    expected = {'config1': ['A', 'B', 'C'], 'config2': ['X', 'Y']}

    assert instrument.possible_options_for_model('mock_v3') == expected


def test_possible_options_for_model_and_configuration(instrument):
    actual = instrument.possible_options_for_model_and_configuration('mock_v3', 'config1')
    assert actual == ['A', 'B', 'C']


def test_get_options(data):
    data = data['version']['VERSION1']['models']['mock_v3']['configurations']['config1']

    assert i.Instrument._get_options(data) == ['A', 'B', 'C']


def test_default_option_for_configuration(instrument):
    assert instrument.default_option_for_configuration('mock_v3', 'config1') == 'A'