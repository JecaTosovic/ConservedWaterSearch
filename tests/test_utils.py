import tempfile

import nglview
import numpy as np
import numpy.testing as npt
import pytest

from ConservedWaterSearch.utils import (
    _append_new_result,
    read_results,
    visualise_nglview,
    visualise_pymol,
)
from ConservedWaterSearch.water_clustering import WaterClustering


def _write_sample_results_file(path, with_hydrogens=True):
    wc = WaterClustering(10, water_types_to_find=["FCW"] if with_hydrogens else ["onlyO"])
    wc._water_type = ["FCW", "HCW"] if with_hydrogens else ["O_clust", "O_clust"]
    wc._waterO = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 1.0]),
    ]
    if with_hydrogens:
        wc._waterH1 = [
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 1.1, 1.0]),
        ]
        wc._waterH2 = [
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([1.0, 1.0, 1.1]),
        ]
    wc.save_results(path)


@pytest.mark.parametrize(
    ("water_type", "waterO", "waterH1", "waterH2", "expected"),
    [
        ("FCW", [1, 2, 3], [4, 5, 6], [7, 8, 9], "FCW 1 2 3 4 5 6 7 8 9\n"),
        ("HCW", [10, 11, 12], None, None, "HCW 10 11 12\n"),
    ],
)
def test_append_new_result(water_type, waterO, waterH1, waterH2, expected):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        _append_new_result(water_type, waterO, waterH1, waterH2, f.name)

        # Ensure file pointer is at the beginning
        f.seek(0)
        contents = f.read()

        assert contents == expected


def test_append_multiple_results():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        # First append
        _append_new_result("FCW", [1, 2, 3], [4, 5, 6], [7, 8, 9], f.name)
        # Second append
        _append_new_result("HCW", [10, 11, 12], None, None, f.name)

        # Ensure file pointer is at the beginning
        f.seek(0)
        contents = f.read()
        expected = "FCW 1 2 3 4 5 6 7 8 9\nHCW 10 11 12\n"
        assert contents == expected


# Fixture for reading results from different files
@pytest.fixture(params=[True, False])
def water_results(request):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        _write_sample_results_file(f.name, with_hydrogens=request.param)
        water_type, waterO, waterH1, waterH2 = read_results(f.name)
    return water_type, waterO, waterH1, waterH2, request.param


# Parametrized test to handle both scenarios
def test_read_results(water_results):
    water_type, waterO, waterH1, waterH2, has_hydrogens = water_results
    assert len(water_type) == 2
    assert len(waterO) == 2
    assert len(waterH1) == 2
    assert len(waterH2) == 2

    # Expected first type based on whether the dataset includes hydrogens
    expected_first_type = "FCW" if has_hydrogens else "O_clust"
    assert water_type[0] == expected_first_type

    # Expected coordinates of waterO
    expected_waterO = np.array([0.0, 0.0, 0.0])
    npt.assert_allclose(waterO[0], expected_waterO, atol=1e-6)

    if has_hydrogens:
        # Expected coordinates for waterH1 and waterH2 if hydrogens are present
        expected_waterH1 = np.array([1.0, 0.0, 0.0])
        expected_waterH2 = np.array([0.0, 1.0, 0.0])
        npt.assert_allclose(waterH1[0], expected_waterH1, atol=1e-6)
        npt.assert_allclose(waterH2[0], expected_waterH2, atol=1e-6)
    else:
        # Ensure waterH1 and waterH2 are empty if hydrogens are not present
        assert waterH1[0] == []
        assert waterH2[0] == []


@pytest.mark.usefixtures("_pymol_skip")
def test_visualise_pymol():
    # Create a temporary file using with for writing and reading
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pse", delete=True) as f:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res_file:
            _write_sample_results_file(res_file.name, with_hydrogens=True)
            tip_res, resO, resH1, resH2 = read_results(res_file.name)
        tip_res[0] = "WCW"
        tip_res[1] = "HCW"
        visualise_pymol(
            tip_res,
            resO,
            resH1,
            resH2,
            aligned_protein=None,
            output_file=f.name,
            lunch_pymol=False,
        )


@pytest.mark.usefixtures("_pymol_skip")
def test_visualise_pymol2():
    # Create a temporary file using with for writing and reading
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=True) as f:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res_file:
            _write_sample_results_file(res_file.name, with_hydrogens=True)
            visualise_pymol(
                *read_results(res_file.name),
                aligned_protein=None,
                output_file=f.name,
                lunch_pymol=False,
            )


# Visualise pymol tests
@pytest.mark.usefixtures("_pymol_skip")
def test_visualise_pymol_with_align_protein():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res_file:
        _write_sample_results_file(res_file.name, with_hydrogens=True)
        visualise_pymol(
            *read_results(res_file.name),
            aligned_protein=None,
            lunch_pymol=False,
        )


def test_visualise_nglview():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res_file:
        _write_sample_results_file(res_file.name, with_hydrogens=True)
        vv = visualise_nglview(
            *read_results(res_file.name),
            aligned_protein=None,
        )
        assert isinstance(vv, nglview.NGLWidget)
