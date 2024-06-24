import tempfile

import numpy as np
import pytest

from ConservedWaterSearch.water_clustering import WaterClustering


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


@pytest.fixture()
def _pymol_skip():
    pytest.importorskip("pymol")


@pytest.fixture(params=[{"onlyO": False}, {"onlyO": True}])
def water_clustering_setup(request):
    wc = WaterClustering(10)
    # Common setup for both cases
    wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
    wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))

    if request.param["onlyO"]:
        # Only oxygen scenario
        wc._water_type.append("O_clust")
        wc._water_type.append("O_clust")
    else:
        # Full water type clustering
        wc._waterH1.append(np.asarray([1.0, 0.0, 0.0]))
        wc._waterH1.append(np.asarray([0.0, 0.8, 0.5]))
        wc._waterH2.append(np.asarray([2.5, 2.0, 2.8]))
        wc._waterH2.append(np.asarray([0.0, 3.0, 0.0]))
        wc._water_type.append("FCW")
        wc._water_type.append("HCW")

    return wc, request.param["onlyO"]


@pytest.fixture()
def water_clustering_setup_for_deletion():
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as dat, tempfile.NamedTemporaryFile(mode="w+", delete=True) as res:
        wc = WaterClustering(10, output_file=res.name, restart_data_file=dat.name)
        Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
        H1 = np.asarray([[0.8, 0.5, 0.8], [1.4, 0.6, 3.7], [3.9, 5.1, 5.9]])
        H2 = np.asarray([[0.4, 0.7, 0.1], [1.7, 3.6, 2.7], [1.8, 3.8, 5.1]])
        yield wc, dat.name, Odata, H1, H2


@pytest.fixture()
def water_clustering_data():
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    return Opos, Hpos
