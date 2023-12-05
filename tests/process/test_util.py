"""Test utility functions"""
import pytest

from crnsynth.process import util


def test_parse_filename_args():
    """Test parse_filename_args"""

    filename = "adult-minimal_privbayes-dk_eps1.0_generalized"
    assert util.parse_filename_args(filename, "dataset_name") == "adult-minimal"
    assert util.parse_filename_args(filename, "generator_name") == "privbayes-dk"
    assert util.parse_filename_args(filename, "epsilon") == 1.0
    assert util.parse_filename_args(filename, "suffix") == "generalized"

    filename_extension = filename + ".csv"
    assert (
        util.parse_filename_args(filename_extension, "dataset_name", extension=".csv")
        == "adult-minimal"
    )
    assert (
        util.parse_filename_args(filename_extension, "generator_name", extension=".csv")
        == "privbayes-dk"
    )
    assert (
        util.parse_filename_args(filename_extension, "epsilon", extension=".csv") == 1.0
    )
    assert (
        util.parse_filename_args(filename_extension, "suffix", extension=".csv")
        == "generalized"
    )

    with pytest.raises(ValueError):
        util.parse_filename_args(filename, "invalid_arg_type")
