import numpy as np
import pytest


@pytest.fixture()
def orientations_normalized():
    return np.asarray([[1, 0, 0], [-0.25038 * 2, 0.96814764 * 2, 0]])


@pytest.fixture()
def orientations_not_normalized():
    return np.asarray([[1, 0, 0], [-0.25038, 0.96814764, 0]])


@pytest.fixture(
    params=[
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
    ]
)
def water_data(request):
    return np.loadtxt(request.param), request.param.split("/")[-1].split(".")[0]


@pytest.fixture(autouse=True)
def _pymol_skip():
    pytest.importorskip("pymol")
