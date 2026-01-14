import re
import pytest

from resins.models.model_base import InstrumentModel, InvalidInputError


class MockModel(InstrumentModel):
    def get_characteristics(self, *args):
        return {}

    def __call__(self, *args, **kwargs):
        return

    def get_kernel(self, points, *meshes):
        return

    def get_peak(self, points, *meshes):
        return

    def broaden(self, points, data, *meshes):
        return


class MockModelData:
    function = ''
    citation = ['']

    def __init__(self, defaults, restrictions):
        self.defaults = defaults
        self.restrictions = restrictions


@pytest.fixture(scope='module')
def model():
    class MockData:
        citation = ['']

    return MockModel(MockData())


def test_validate_settings_default(model):
    default = 100
    model_data = MockModelData({'setting': default}, {})

    result = model._validate_settings(model_data, {'setting': None})
    assert result['setting'] == default


def test_validate_settings_no_default(model):
    model_data = MockModelData({}, {})

    with pytest.raises(InvalidInputError,
                       match='Model "MockModel" does not have a default value for the "no_default"'):
        model._validate_settings(model_data, {'no_default': None})


def test_validate_settings_default_skips_restrictions(model):
    default = 100
    model_data = MockModelData({'setting': default}, {'setting': [0, 1]})

    result = model._validate_settings(model_data, {'setting': None})
    assert result['setting'] == default


def test_validate_settings_default_not_applied_when_value(model):
    value = 100
    model_data = MockModelData({'setting': 333}, {})

    result = model._validate_settings(model_data, {'setting': value})
    assert result['setting'] == value


def test_validate_settings_restriction_list2_works(model):
    value = 50
    model_data = MockModelData({}, {'setting': [0, 100]})

    result = model._validate_settings(model_data, {'setting': value})
    assert result['setting'] == value


def test_validate_settings_restriction_list2_raises(model):
    model_data = MockModelData({}, {'setting': [0, 100]})

    with pytest.raises(InvalidInputError,
                       match=re.escape('The provided value for the "setting" setting (500) must be '
                                       'within the [0, 100] boundaries.')):
        model._validate_settings(model_data, {'setting': 500})


def test_validate_settings_restriction_list3_works(model):
    value = 200
    model_data = MockModelData({}, {'setting': [100, 500, 100]})

    result = model._validate_settings(model_data, {'setting': value})
    assert result['setting'] == value


def test_validate_settings_restriction_list3(model):
    model_data = MockModelData({}, {'setting': [100, 500, 100]})

    with pytest.raises(InvalidInputError,
                       match=re.escape('The provided value for the "setting" setting (500) must be '
                                       'one of the following values: [100, 200, 300, 400]')):
        model._validate_settings(model_data, {'setting': 500})


def test_validate_settings_restriction_list_wrong_length(model):
    model_data = MockModelData({}, {'setting': [100]})

    with pytest.raises(ValueError):
        model._validate_settings(model_data, {'setting': 500})


def test_validate_settings_restriction_set_works(model):
    value = 200
    model_data = MockModelData({}, {'setting': {100, 200, 300, 400}})

    result = model._validate_settings(model_data, {'setting': value})
    assert result['setting'] == value


def test_validate_settings_restriction_set_raises(model):
    model_data = MockModelData({}, {'setting': {100, 200, 300, 400}})

    with pytest.raises(InvalidInputError,
                       match=r'The provided value for the \"setting\" setting \(500\) must be one '
                             r'of the following values: {.+}'):
        model._validate_settings(model_data, {'setting': 500})
