"""Unit and regression test for the ConservedWaterSearch package."""

# Import package, test suite, and other packages as needed
import numpy as np
import pytest

import ConservedWaterSearch.hydrogen_orientation

make_ho_plots = 0


@pytest.mark.parametrize(
    "data_file",
    [
        "tests/data/conserved_sample_FCW.dat",
        "tests/data/conserved_sample_FCW2.dat",
        "tests/data/dispersed_sample_HCW.dat",
        "tests/data/dispersed_sample_HCW2.dat",
        "tests/data/circ_sample_HCW.dat",
        "tests/data/circ_sample_HCW2.dat",
        "tests/data/circ_sample_HCW3.dat",
        "tests/data/sample_WCW.dat",
        "tests/data/sample_circular_only_WCW.dat",
        "tests/data/dispersed_sample_WCW.dat",
        "tests/data/2_by_2_WCW.dat",
        "tests/data/dispersed_sample_not_conserved.dat",
        "tests/data/not_conserved.dat",
    ],
)
def test_water_types_from_files(data_file):
    orientations = np.loadtxt(data_file)
    data_label = data_file.split("/")[-1].split(".")[0]
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations, debugH=make_ho_plots
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    if "not_conserved" in data_label:
        assert len(res) == 0
    else:
        assert len(res) > 0
        assert res[0][2] in data_label.upper()


def test_water_types(water_data):
    orientations, expected = water_data
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations, debugH=make_ho_plots
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    if expected is None:
        assert len(res) == 0
    else:
        assert len(res) > 0
        if isinstance(expected, set):
            assert res[0][2] in expected
        else:
            assert res[0][2] == expected


@pytest.mark.parametrize(
    ("orientations", "expected"),
    [
        (
            np.asarray([[[1]]]),
            pytest.raises(ValueError, match="Orientations have to be a 2D array"),
        ),
        (
            np.asarray([[1, 0], [0, 1]]),
            pytest.raises(
                ValueError, match="Orientations must be vectors of dimension 3"
            ),
        ),
        (
            np.asarray([[1, 0, 0]]),
            pytest.raises(
                ValueError,
                match=(
                    "Number of orientations must be even!"
                    " Each water molecule has 2 hydrogen atoms!"
                ),
            ),
        ),
        (
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            pytest.raises(
                ValueError,
                match=(
                    "Number of orientations must be even!"
                    " Each water molecule has 2 hydrogen atoms!"
                ),
            ),
        ),
    ],
)
def test_invalid_orientations(orientations, expected):
    with expected:
        ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
            orientations
        )


@pytest.mark.parametrize("normalize_orientations", [True, False])
def test_orientation_normalization(
    orientations_normalized, orientations_not_normalized, normalize_orientations
):
    w1 = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations_normalized
        if normalize_orientations
        else orientations_not_normalized
    )
    w2 = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations_not_normalized, normalize_orientations=normalize_orientations
    )
    assert w1 == w2


def test_orientation_valid_input():
    orientations = np.asarray([[1, 0, 0], [0, 1, 0]])
    ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations
    )
