import os

import numpy as np
import numpy.testing as npt

from ConservedWaterSearch.utils import (
    get_orientations_from_positions,
    read_results,
)
from ConservedWaterSearch.water_clustering import WaterClustering


def test_restoring_defaults() -> None:
    wc = WaterClustering(10)
    wc.restore_default_options()
    wc2 = WaterClustering(10)
    d1 = wc.__dict__
    d2 = wc2.__dict__
    for i in d1.keys():
        assert d1[i] == d2[i]


def test_restoring_defaults2():
    wc = WaterClustering(10)
    wc.debugH = 5
    wc.halfcon_angstd_cutoff = 0.6666
    wc._water_type = ["FCW"]
    wc.restore_default_options(delete_results=True)
    wc2 = WaterClustering(10)
    d1 = wc.__dict__
    d2 = wc2.__dict__
    for i in d1.keys():
        assert d1[i] == d2[i]


def test_save_results1():
    wc = WaterClustering(10)
    wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
    wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
    wc._water_type.append("O_clust")
    wc._water_type.append("O_clust")
    wc.save_results()
    a, b, c, d = read_results()
    for i, j in zip(a, wc.water_type):
        assert i == j
    for i, j in zip(b, wc.waterO):
        npt.assert_allclose(i, j)
    os.remove("Clustering_results.dat")
    os.remove("Type_Clustering_results.dat")


def test_save_results2():
    wc = WaterClustering(10)
    wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
    wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
    wc._waterH1.append(np.asarray([1.0, 0.0, 0.0]))
    wc._waterH1.append(np.asarray([0.0, 0.8, 0.5]))
    wc._waterH2.append(np.asarray([2.5, 2.0, 2.8]))
    wc._waterH2.append(np.asarray([0.0, 3.0, 0.0]))
    wc._water_type.append("FCW")
    wc._water_type.append("FCW")
    wc.save_results()
    a, b, c, d = read_results()
    for i, j in zip(a, wc.water_type):
        assert i == j
    for i, j in zip(b, wc.waterO):
        npt.assert_allclose(i, j)
    for i, j in zip(c, wc.waterH1):
        npt.assert_allclose(i, j)
    for i, j in zip(d, wc.waterH2):
        npt.assert_allclose(i, j)
    os.remove("Clustering_results.dat")
    os.remove("Type_Clustering_results.dat")


def test_save_clustering_options_and_create_from_file():
    wc = WaterClustering(10)
    ct = "multi_stage_reclustering"
    ca = "OPTICS"
    options = [0.5, 1, [0.05, 0.01, 0.001]]
    whichH = ["onlyO"]
    wc.save_clustering_options(
        clustering_type=ct,
        clustering_algorithm=ca,
        options=options,
        whichH=whichH,
    )
    newWC = WaterClustering.create_from_file()
    d1 = wc.__dict__
    d2 = newWC.__dict__
    for i in d1.keys():
        assert d1[i] == d2[i]
    os.remove("clust_options.dat")


def test_read_clustering_options():
    wc = WaterClustering(10)
    ct = "multi_stage_reclustering"
    ca = "OPTICS"
    options = [0.5, 1, [0.05, 0.01, 0.001]]
    whichH = ["onlyO"]
    wc.save_clustering_options(
        clustering_type=ct,
        clustering_algorithm=ca,
        options=options,
        whichH=whichH,
    )
    newWC = WaterClustering(0)
    newWC.read_class_options()
    d1 = wc.__dict__
    d2 = newWC.__dict__
    for i in d1.keys():
        assert d1[i] == d2[i]
    os.remove("clust_options.dat")


def test_read_water_clust_options():
    wc = WaterClustering(10)
    ct = "multi_stage_reclustering"
    ca = "OPTICS"
    options = [0.5, 1, [0.05, 0.01, 0.001]]
    whichH = ["onlyO"]
    wc.save_clustering_options(
        clustering_type=ct,
        clustering_algorithm=ca,
        options=options,
        whichH=whichH,
    )
    newWC = WaterClustering(0)
    newclstype, newclsalg, newop, newwhich = newWC.read_water_clust_options()
    assert newclstype == ct
    assert newclsalg == ca
    assert newop == options
    assert newwhich == whichH
    os.remove("clust_options.dat")


def test_restart_cluster():
    wc = WaterClustering(10)
    wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
    wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
    wc._waterH1.append(np.asarray([1.0, 0.0, 0.0]))
    wc._waterH1.append(np.asarray([0.0, 0.8, 0.5]))
    wc._waterH2.append(np.asarray([2.5, 2.0, 2.8]))
    wc._waterH2.append(np.asarray([0.0, 3.0, 0.0]))
    wc._water_type.append("FCW")
    wc._water_type.append("FCW")
    wc.save_results()
    ct = "multi_stage_reclustering"
    ca = "OPTICS"
    options = [0.5, 1, [0.05, 0.01, 0.001]]
    whichH = ["onlyO"]
    wc.save_clustering_options(
        clustering_type=ct,
        clustering_algorithm=ca,
        options=options,
        whichH=whichH,
    )
    Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
    np.savetxt("water_coords_restart.dat", np.c_[Odata])
    newWC = WaterClustering(0)
    newWC.save_intermediate_results = False
    newWC.restart_cluster(
        results_file="Clustering_results.dat",
        type_results_file="Type_Clustering_results.dat",
    )
    os.remove("clust_options.dat")
    os.remove("Clustering_results.dat")
    os.remove("Type_Clustering_results.dat")


def test_delete_data():
    wc = WaterClustering(0)
    Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
    Onew, _, _ = wc._delete_data([0, 2], Odata=Odata)
    npt.assert_allclose(Onew[0], Odata[1])
    assert os.path.isfile("water_coords_restart.dat")
    os.remove("water_coords_restart.dat")


def test_delete_data2():
    wc = WaterClustering(0)
    Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
    wc.save_intermediate_results = False
    wc._delete_data([0, 2], Odata=Odata)
    assert not (os.path.isfile("water_coords_restart.dat"))


def test_delete_data3():
    wc = WaterClustering(0)
    Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
    H1 = np.asarray([[0.8, 0.5, 0.8], [1.4, 0.6, 3.7], [3.9, 5.1, 5.9]])
    H2 = np.asarray([[0.4, 0.7, 0.1], [1.7, 3.6, 2.7], [1.8, 3.8, 5.1]])
    Onew, H1new, H2new = wc._delete_data([1, 2], Odata, H1, H2)
    npt.assert_allclose(Onew[0], Odata[0])
    npt.assert_allclose(H1new[0], H1[0])
    npt.assert_allclose(H2new[0], H2[0])
    assert os.path.isfile("water_coords_restart.dat")
    os.remove("water_coords_restart.dat")


def test_single_clustering_OPTICS():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(
        Nsnap, save_intermediate_results=False, save_results_after_done=False
    )
    wc.single_clustering(*get_orientations_from_positions(Opos, Hpos))
    a, b, c, d = read_results(
        "tests/data/Single_OPTICS.dat", "tests/data/Single_OPTICS_Type.dat"
    )
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_single_clustering_HDBSCAN():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(
        Nsnap, save_intermediate_results=False, save_results_after_done=False
    )
    wc.single_clustering(
        *get_orientations_from_positions(Opos, Hpos), clustering_algorithm="HDBSCAN"
    )
    a, b, c, d = read_results(
        "tests/data/Single_HDBSCAN.dat", "tests/data/Single_HDBSCAN_Type.dat"
    )
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_multistage_reclustering_OPTICS():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(
        Nsnap, save_intermediate_results=False, save_results_after_done=False
    )
    wc.multi_stage_reclustering(*get_orientations_from_positions(Opos, Hpos))
    a, b, c, d = read_results(
        "tests/data/MSR_OPTICS.dat", "tests/data/MSR_OPTICS_Type.dat"
    )
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_multistage_reclustering_HDBSCAN():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(
        Nsnap, save_intermediate_results=False, save_results_after_done=False
    )
    wc.multi_stage_reclustering(
        *get_orientations_from_positions(Opos, Hpos), clustering_algorithm="HDBSCAN"
    )
    a, b, c, d = read_results(
        "tests/data/MSR_HDBSCAN.dat", "tests/data/MSR_HDBSCAN_Type.dat"
    )
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))
