import os
import shutil

import nglview
import numpy as np
import numpy.testing as npt

from ConservedWaterSearch.utils import (
    read_results,
    visualise_nglview,
    visualise_pymol,
)


def test_visualise_pymol():
    tip_res, resO, resH1, resH2 = read_results(
        "tests/data/Clustering_results.dat",
        "tests/data/Type_Clustering_results.dat",
    )
    tip_res[2] = "WCW"
    tip_res[4] = "HCW"
    visualise_pymol(
        tip_res,
        resO,
        resH1,
        resH2,
        aligned_protein=None,
        output_file="test.pse",
    )
    os.remove("test.pse")


def test_visualise_pymol2():
    visualise_pymol(
        *read_results(
            "tests/data/Clustering_results.dat",
            "tests/data/Type_Clustering_results.dat",
        ),
        aligned_protein=None,
        output_file="test.pdb",
        crystal_waters="3T74",
        ligand_resname="UBY",
    )
    os.remove("test.pdb")


def test_visualise_pymol3():
    visualise_pymol(
        *read_results(
            "tests/data/Clustering_results.dat",
            "tests/data/Type_Clustering_results.dat",
        ),
        aligned_protein=None,
    )


def test_visualise_pymol4():
    visualise_pymol(
        *read_results(
            "tests/data/Clustering_results.dat",
            "tests/data/Type_Clustering_results.dat",
        ),
        aligned_protein="tests/data/aligned.pdb",
    )


def test_visualise_nglview():
    vv = visualise_nglview(
        *read_results(
            "tests/data/Clustering_results.dat",
            "tests/data/Type_Clustering_results.dat",
        ),
        aligned_protein=None,
        crystal_waters="3T74",
    )
    assert type(vv) == nglview.NGLWidget


def test_read_results():
    fle = "tests/data/Clustering_results.dat"
    typefle = "tests/data/Type_Clustering_results.dat"
    water_type, waterO, waterH1, waterH2 = read_results(fle, typefle)
    data = np.loadtxt(fle)
    new_waterO = data[:, :3]
    new_waterH1 = data[:, 3:6]
    new_waterH2 = data[:, 6:9]
    assert water_type == ["FCW"] * 20
    npt.assert_allclose(new_waterO, waterO)
    npt.assert_allclose(new_waterH1, waterH1)
    npt.assert_allclose(new_waterH2, waterH2)


def test_read_results2():
    fle = "tests/data/Clustering_results.dat"
    typefle = "tests/data/Type_Clustering_results.dat"
    newfle = "Clustering_results.dat"
    newtypefle = "Type_Clustering_results.dat"
    water_type, waterO, waterH1, waterH2 = read_results(fle, typefle)
    shutil.copy(fle, newfle)
    shutil.copy(typefle, newtypefle)
    newwater_type, new_waterO, new_waterH1, new_waterH2 = read_results(
        newfle, newtypefle
    )
    os.remove(newfle)
    os.remove(newtypefle)
    assert water_type == newwater_type
    npt.assert_allclose(new_waterO, waterO)
    npt.assert_allclose(new_waterH1, waterH1)
    npt.assert_allclose(new_waterH2, waterH2)
