import sys
sys.path.append('/Users/bensolis-cohen/Projects/alertme')
from codegen import check_code


import pytest

class TestCheckCode:

    # Correctly compares matching JSON outputs
    def test_correctly_compares_matching_json_outputs(self):
        global_output = '{"key": "value"}'
        expected_output = '{"key": "value"}'
        assert check_code(global_output, expected_output) == True

    # Handles invalid JSON in global_output gracefully
    def test_handles_invalid_json_in_global_output(self):
        global_output = 'invalid json'
        expected_output = '{"key": "value"}'
        assert check_code(global_output, expected_output) == False
        
