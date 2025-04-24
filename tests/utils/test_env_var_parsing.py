# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from datetime import date, datetime
from decimal import Decimal
from unittest import mock

import pytest

from nemo.utils.env_var_parsing import (
    CoercionError,
    RequiredSettingMissingError,
    get_envbool,
    get_envdate,
    get_envdatetime,
    get_envdecimal,
    get_envdict,
    get_envfloat,
    get_envint,
    get_envlist,
)


class TestEnvironmentVariableParsing:

    def test_get_envint_returns_int_value(self):
        """Test that get_envint returns the integer value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_INT': '42'}):
            assert get_envint('TEST_INT') == 42

    def test_get_envint_with_default(self):
        """Test that get_envint returns the default value when env var is missing."""
        assert get_envint('NONEXISTENT_INT', 123) == 123

    def test_get_envint_required_missing(self):
        """Test that get_envint raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envint('NONEXISTENT_INT')

    def test_get_envint_coercion_error(self):
        """Test that get_envint raises a CoercionError for non-integer values."""
        with mock.patch.dict(os.environ, {'TEST_INT': 'not-an-int'}):
            with pytest.raises(CoercionError):
                get_envint('TEST_INT')

    def test_get_envint_negative_value(self):
        """Test that get_envint correctly handles negative integers."""
        with mock.patch.dict(os.environ, {'TEST_INT': '-42'}):
            assert get_envint('TEST_INT') == -42

    def test_get_envfloat_returns_float_value(self):
        """Test that get_envfloat returns the float value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_FLOAT': '3.14'}):
            assert get_envfloat('TEST_FLOAT') == 3.14

    def test_get_envfloat_with_integer_string(self):
        """Test that get_envfloat correctly converts integer strings to floats."""
        with mock.patch.dict(os.environ, {'TEST_FLOAT': '42'}):
            value = get_envfloat('TEST_FLOAT')
            assert value == 42.0
            assert isinstance(value, float)

    def test_get_envfloat_with_default(self):
        """Test that get_envfloat returns the default value when env var is missing."""
        assert get_envfloat('NONEXISTENT_FLOAT', 3.14) == 3.14

    def test_get_envfloat_required_missing(self):
        """Test that get_envfloat raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envfloat('NONEXISTENT_FLOAT')

    def test_get_envfloat_coercion_error(self):
        """Test that get_envfloat raises a CoercionError for non-float values."""
        with mock.patch.dict(os.environ, {'TEST_FLOAT': 'not-a-float'}):
            with pytest.raises(CoercionError):
                get_envfloat('TEST_FLOAT')

    def test_get_envfloat_scientific_notation(self):
        """Test that get_envfloat correctly handles scientific notation."""
        with mock.patch.dict(os.environ, {'TEST_FLOAT': '1.23e-4'}):
            assert get_envfloat('TEST_FLOAT') == 1.23e-4

    def test_get_envfloat_negative_value(self):
        """Test that get_envfloat correctly handles negative values."""
        with mock.patch.dict(os.environ, {'TEST_FLOAT': '-3.14'}):
            assert get_envfloat('TEST_FLOAT') == -3.14

    # Tests for get_envbool
    def test_get_envbool_true_values(self):
        """Test that get_envbool returns True for various truthy values."""
        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES', 'y', 'Y', 't', 'T']
        for val in true_values:
            with mock.patch.dict(os.environ, {'TEST_BOOL': val}):
                assert get_envbool('TEST_BOOL') is True

    def test_get_envbool_false_values(self):
        """Test that get_envbool returns False for various falsy values."""
        false_values = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', 'n', 'N', 'f', 'F', 'none', 'None', 'NONE']
        for val in false_values:
            with mock.patch.dict(os.environ, {'TEST_BOOL': val}):
                assert get_envbool('TEST_BOOL') is False

    def test_get_envbool_with_default(self):
        """Test that get_envbool returns the default value when env var is missing."""
        assert get_envbool('NONEXISTENT_BOOL', True) is True
        assert get_envbool('NONEXISTENT_BOOL', False) is False

    def test_get_envbool_required_missing(self):
        """Test that get_envbool raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envbool('NONEXISTENT_BOOL')

    def test_get_envbool_non_boolean_value(self):
        """Test that get_envbool interprets non-standard values as True."""
        with mock.patch.dict(os.environ, {'TEST_BOOL': 'something-else'}):
            assert get_envbool('TEST_BOOL') is True

    # Tests for get_envdecimal
    def test_get_envdecimal_returns_decimal_value(self):
        """Test that get_envdecimal returns the Decimal value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_DECIMAL': '3.14'}):
            value = get_envdecimal('TEST_DECIMAL')
            assert value == Decimal('3.14')
            assert isinstance(value, Decimal)

    def test_get_envdecimal_with_integer_string(self):
        """Test that get_envdecimal correctly converts integer strings to Decimals."""
        with mock.patch.dict(os.environ, {'TEST_DECIMAL': '42'}):
            value = get_envdecimal('TEST_DECIMAL')
            assert value == Decimal('42')
            assert isinstance(value, Decimal)

    def test_get_envdecimal_with_default(self):
        """Test that get_envdecimal returns the default value when env var is missing."""
        default_value = Decimal('3.14')
        assert get_envdecimal('NONEXISTENT_DECIMAL', default_value) == default_value

    def test_get_envdecimal_required_missing(self):
        """Test that get_envdecimal raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envdecimal('NONEXISTENT_DECIMAL')

    def test_get_envdecimal_coercion_error(self):
        """Test that get_envdecimal raises a CoercionError for non-decimal values."""
        with mock.patch.dict(os.environ, {'TEST_DECIMAL': 'not-a-decimal'}):
            with pytest.raises(CoercionError):
                get_envdecimal('TEST_DECIMAL')

    def test_get_envdecimal_negative_value(self):
        """Test that get_envdecimal correctly handles negative values."""
        with mock.patch.dict(os.environ, {'TEST_DECIMAL': '-3.14'}):
            assert get_envdecimal('TEST_DECIMAL') == Decimal('-3.14')

    def test_get_envdecimal_high_precision(self):
        """Test that get_envdecimal preserves high precision values."""
        with mock.patch.dict(os.environ, {'TEST_DECIMAL': '3.1415926535897932384626433832795028841971'}):
            value = get_envdecimal('TEST_DECIMAL')
            assert value == Decimal('3.1415926535897932384626433832795028841971')

    # Tests for get_envdate
    def test_get_envdate_returns_date_value(self):
        """Test that get_envdate returns the date value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_DATE': '2023-05-15'}):
            value = get_envdate('TEST_DATE')
            assert value == date(2023, 5, 15)
            assert isinstance(value, date)

    def test_get_envdate_with_different_formats(self):
        """Test that get_envdate handles different date formats."""
        date_formats = {
            '2023-05-15': date(2023, 5, 15),
            '15-05-2023': date(2023, 5, 15),
            '05/15/2023': date(2023, 5, 15),
            '15 May 2023': date(2023, 5, 15),
            'May 15, 2023': date(2023, 5, 15),
        }

        for date_str, expected_date in date_formats.items():
            with mock.patch.dict(os.environ, {'TEST_DATE': date_str}):
                assert get_envdate('TEST_DATE') == expected_date

    def test_get_envdate_with_default(self):
        """Test that get_envdate returns the default value when env var is missing."""
        default_value = date(2023, 5, 15)
        assert get_envdate('NONEXISTENT_DATE', default_value) == default_value

    def test_get_envdate_required_missing(self):
        """Test that get_envdate raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envdate('NONEXISTENT_DATE')

    def test_get_envdate_coercion_error(self):
        """Test that get_envdate raises a CoercionError for invalid date values."""
        with mock.patch.dict(os.environ, {'TEST_DATE': 'not-a-date'}):
            with pytest.raises(CoercionError):
                get_envdate('TEST_DATE')

    # Tests for get_envdatetime
    def test_get_envdatetime_returns_datetime_value(self):
        """Test that get_envdatetime returns the datetime value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_DATETIME': '2023-05-15T14:30:45'}):
            value = get_envdatetime('TEST_DATETIME')
            assert value == datetime(2023, 5, 15, 14, 30, 45)
            assert isinstance(value, datetime)

    def test_get_envdatetime_with_different_formats(self):
        """Test that get_envdatetime handles different datetime formats."""
        datetime_formats = {
            '2023-05-15T14:30:45': datetime(2023, 5, 15, 14, 30, 45),
            '2023-05-15 14:30:45': datetime(2023, 5, 15, 14, 30, 45),
            '15-05-2023 14:30:45': datetime(2023, 5, 15, 14, 30, 45),
            '05/15/2023 14:30:45': datetime(2023, 5, 15, 14, 30, 45),
            '15 May 2023 14:30:45': datetime(2023, 5, 15, 14, 30, 45),
        }

        for datetime_str, expected_datetime in datetime_formats.items():
            with mock.patch.dict(os.environ, {'TEST_DATETIME': datetime_str}):
                assert get_envdatetime('TEST_DATETIME') == expected_datetime

    def test_get_envdatetime_with_default(self):
        """Test that get_envdatetime returns the default value when env var is missing."""
        default_value = datetime(2023, 5, 15, 14, 30, 45)
        assert get_envdatetime('NONEXISTENT_DATETIME', default_value) == default_value

    def test_get_envdatetime_required_missing(self):
        """Test that get_envdatetime raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envdatetime('NONEXISTENT_DATETIME')

    def test_get_envdatetime_coercion_error(self):
        """Test that get_envdatetime raises a CoercionError for invalid datetime values."""
        with mock.patch.dict(os.environ, {'TEST_DATETIME': 'not-a-datetime'}):
            with pytest.raises(CoercionError):
                get_envdatetime('TEST_DATETIME')

    def test_get_envdatetime_with_timezone(self):
        """Test that get_envdatetime handles timezone information."""
        with mock.patch.dict(os.environ, {'TEST_DATETIME': '2023-05-15T14:30:45+0200'}):
            dt = get_envdatetime('TEST_DATETIME')
            assert dt.year == 2023
            assert dt.month == 5
            assert dt.day == 15
            assert dt.hour == 14
            assert dt.minute == 30
            assert dt.second == 45

    # Tests for get_envlist
    def test_get_envlist_returns_list_value(self):
        """Test that get_envlist returns the list value from environment variable."""
        with mock.patch.dict(os.environ, {'TEST_LIST': 'item1 item2 item3'}):
            value = get_envlist('TEST_LIST')
            assert value == ['item1', 'item2', 'item3']
            assert isinstance(value, list)

    def test_get_envlist_with_custom_separator(self):
        """Test that get_envlist handles custom separators."""
        with mock.patch.dict(os.environ, {'TEST_LIST': 'item1,item2,item3'}):
            value = get_envlist('TEST_LIST', separator=',')
            assert value == ['item1', 'item2', 'item3']

    def test_get_envlist_with_default(self):
        """Test that get_envlist returns the default value when env var is missing."""
        default_value = ['default1', 'default2']
        assert get_envlist('NONEXISTENT_LIST', default_value) == default_value

    def test_get_envlist_required_missing(self):
        """Test that get_envlist raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envlist('NONEXISTENT_LIST')

    def test_get_envlist_empty_string(self):
        """Test that get_envlist handles empty strings."""
        with mock.patch.dict(os.environ, {'TEST_LIST': ''}):
            value = get_envlist('TEST_LIST')
            assert value == ['']

    def test_get_envlist_multiple_words(self):
        """Test that get_envlist correctly splits words with spaces."""
        with mock.patch.dict(os.environ, {'TEST_LIST': 'word1 "phrase with spaces" word3'}):
            value = get_envlist('TEST_LIST')
            assert value == ['word1', '"phrase', 'with', 'spaces"', 'word3']

    # Tests for get_envdict
    def test_get_envdict_returns_dict_value(self):
        """Test that get_envdict returns the dict value from environment variable."""
        test_dict = {'key1': 'value1', 'key2': 42, 'key3': True}
        with mock.patch.dict(os.environ, {'TEST_DICT': json.dumps(test_dict)}):
            value = get_envdict('TEST_DICT')
            assert value == test_dict
            assert isinstance(value, dict)

    def test_get_envdict_with_default(self):
        """Test that get_envdict returns the default value when env var is missing."""
        default_value = {'default_key': 'default_value'}
        assert get_envdict('NONEXISTENT_DICT', default_value) == default_value

    def test_get_envdict_required_missing(self):
        """Test that get_envdict raises an exception when required env var is missing."""
        with pytest.raises(RequiredSettingMissingError):
            get_envdict('NONEXISTENT_DICT')

    def test_get_envdict_coercion_error(self):
        """Test that get_envdict raises a CoercionError for invalid JSON."""
        with mock.patch.dict(os.environ, {'TEST_DICT': 'not-valid-json'}):
            with pytest.raises(CoercionError):
                get_envdict('TEST_DICT')

    def test_get_envdict_complex_dict(self):
        """Test that get_envdict handles complex nested dictionaries."""
        complex_dict = {
            'string': 'value',
            'number': 42,
            'boolean': True,
            'list': [1, 2, 3],
            'nested': {'key1': 'value1', 'key2': [4, 5, 6]},
        }
        with mock.patch.dict(os.environ, {'TEST_DICT': json.dumps(complex_dict)}):
            value = get_envdict('TEST_DICT')
            assert value == complex_dict
