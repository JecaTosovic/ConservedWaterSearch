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
def pymol_skip():
    pytest.importorskip("pymol")


def test_append_new_result():
    # Create a temporary file using with for writing and reading
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        # Call the function with some test data
        _append_new_result("FCW", [1, 2, 3], [4, 5, 6], [7, 8, 9], f.name)

        # Read the contents of the file
        with open(f.name) as file:
            contents = file.read()

        # Check that the contents are as expected
        expected = "FCW 1 2 3 4 5 6 7 8 9\n"
        assert contents == expected

        # Call the function again with different test data
        _append_new_result("HCW", [10, 11, 12], None, None, f.name)

        # Read the contents of the file again
        with open(f.name) as file:
            contents = file.read()

        # Check that the contents are as expected
        expected += "HCW 10 11 12\n"
        assert contents == expected


def test_read_results_with_hydrogens():
    water_type, waterO, waterH1, waterH2 = read_results(
        "tests/data/merged_new_clustering_results.dat",
    )
    assert len(water_type) == 20
    assert len(waterO) == 20
    assert len(waterH1) == 20
    assert len(waterH2) == 20
    assert water_type[0] == "FCW"
    npt.assert_allclose(
        waterO[0],
        np.array(
            [
                -8.498636033210043905e00,
                -8.528824215611816584e00,
                -8.558124911970363513e00,
            ]
        ),
    )
    npt.assert_allclose(
        waterH1[0],
        np.array(
            [
                -8.612760033927649772e00,
                -8.398256579537193289e00,
                -9.542974039943826980e00,
            ]
        ),
    )
    npt.assert_allclose(
        waterH2[0],
        np.array(
            [
                -7.847825970288877961e00,
                -7.856730538101417416e00,
                -8.204944574676375169e00,
            ]
        ),
    )


def test_read_results_without_hydrogens():
    water_type, waterO, waterH1, waterH2 = read_results(
        "tests/data/merged_new_clustering_results_noH.dat",
    )
    assert len(water_type) == 20
    assert len(waterO) == 20
    assert len(waterH1) == 20
    assert len(waterH2) == 20
    assert water_type[0] == "O_clust"
    npt.assert_allclose(
        waterO[0],
        np.array(
            [
                -8.498636033210043905e00,
                -8.528824215611816584e00,
                -8.558124911970363513e00,
            ]
        ),
    )
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


def test_visualise_pymol3():
    visualise_pymol(
        *read_results(
            "tests/data/merged_new_clustering_results.dat",
        ),
        aligned_protein=None,
        lunch_pymol=False,
    )


def test_visualise_pymol4():
    visualise_pymol(
        *read_results(
            "tests/data/merged_new_clustering_results.dat",
        ),
        aligned_protein="tests/data/aligned.pdb",
        lunch_pymol=False,
    )


def test_visualise_pymol5():
    visualise_pymol(
        *read_results(
            "tests/data/merged_new_clustering_results_noH.dat",
        ),
        aligned_protein="tests/data/aligned.pdb",
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
    assert type(vv) is nglview.NGLWidget
