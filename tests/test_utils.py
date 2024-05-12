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


@pytest.fixture(autouse=True)
def _pymol_skip():
    pytest.importorskip("pymol")


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
@pytest.fixture(
    params=[
        {
            "filename": "tests/data/merged_new_clustering_results.dat",
            "has_hydrogens": True,
        },
        {
            "filename": "tests/data/merged_new_clustering_results_noH.dat",
            "has_hydrogens": False,
        },
    ]
)
def water_results(request):
    # Read results from the specified file
    water_type, waterO, waterH1, waterH2 = read_results(request.param["filename"])
    return water_type, waterO, waterH1, waterH2, request.param["has_hydrogens"]


# Parametrized test to handle both scenarios
def test_read_results(water_results):
    water_type, waterO, waterH1, waterH2, has_hydrogens = water_results
    assert len(water_type) == 20
    assert len(waterO) == 20
    assert len(waterH1) == 20
    assert len(waterH2) == 20

    # Expected first type based on whether the dataset includes hydrogens
    expected_first_type = "FCW" if has_hydrogens else "O_clust"
    assert water_type[0] == expected_first_type

    # Expected coordinates of waterO
    expected_waterO = np.array(
        [-8.498636033210043905, -8.528824215611816584, -8.558124911970363513]
    )
    npt.assert_allclose(waterO[0], expected_waterO, atol=1e-6)

    if has_hydrogens:
        # Expected coordinates for waterH1 and waterH2 if hydrogens are present
        expected_waterH1 = np.array(
            [-8.612760033927649772, -8.398256579537193289, -9.542974039943826980]
        )
        expected_waterH2 = np.array(
            [-7.847825970288877961, -7.856730538101417416, -8.204944574676375169]
        )
        npt.assert_allclose(waterH1[0], expected_waterH1, atol=1e-6)
        npt.assert_allclose(waterH2[0], expected_waterH2, atol=1e-6)
    else:
        # Ensure waterH1 and waterH2 are empty if hydrogens are not present
        assert waterH1[0] == []
        assert waterH2[0] == []


def test_visualise_pymol():
    # Create a temporary file using with for writing and reading
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pse", delete=True) as f:
        tip_res, resO, resH1, resH2 = read_results(
            "tests/data/merged_new_clustering_results.dat",
        )
        tip_res[2] = "WCW"
        tip_res[4] = "HCW"
        visualise_pymol(
            tip_res,
            resO,
            resH1,
            resH2,
            aligned_protein=None,
            output_file=f.name,
            lunch_pymol=False,
        )


def test_visualise_pymol2():
    # Create a temporary file using with for writing and reading
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=True) as f:
        visualise_pymol(
            *read_results(
                "tests/data/merged_new_clustering_results.dat",
            ),
            aligned_protein=None,
            output_file=f.name,
            crystal_waters="3T74",
            ligand_resname="UBY",
            lunch_pymol=False,
        )


# Visualise pymol tests
@pytest.mark.parametrize(
    ("output_file", "align_file"),
    [
        ("tests/data/merged_new_clustering_results.dat", None),
        ("tests/data/merged_new_clustering_results.dat", "tests/data/aligned.pdb"),
        ("tests/data/merged_new_clustering_results_noH.dat", "tests/data/aligned.pdb"),
    ],
)
def test_visualise_pymol_with_align_protein(output_file, align_file):
    visualise_pymol(
        *read_results(output_file),
        aligned_protein=align_file,
        lunch_pymol=False,
    )


def test_visualise_nglview():
    vv = visualise_nglview(
        *read_results(
            "tests/data/merged_new_clustering_results.dat",
        ),
        aligned_protein=None,
        crystal_waters="3T74",
    )
    assert isinstance(vv, nglview.NGLWidget)
