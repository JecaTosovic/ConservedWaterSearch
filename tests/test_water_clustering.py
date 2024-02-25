import os

import numpy as np
import numpy.testing as npt
import tempfile

from ConservedWaterSearch.utils import (
    get_orientations_from_positions,
    read_results,
)
from ConservedWaterSearch.water_clustering import WaterClustering


def test_save_results_water_types():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        wc = WaterClustering(10)
        wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
        wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
        wc._waterH1.append(np.asarray([1.0, 0.0, 0.0]))
        wc._waterH1.append(np.asarray([0.0, 0.8, 0.5]))
        wc._waterH2.append(np.asarray([2.5, 2.0, 2.8]))
        wc._waterH2.append(np.asarray([0.0, 3.0, 0.0]))
        wc._water_type.append("FCW")
        wc._water_type.append("HCW")
        wc.save_results(f.name)
        a, b, c, d = read_results(f.name)
        for i, j in zip(a, wc.water_type):
            assert i == j
        for i, j in zip(b, wc.waterO):
            npt.assert_allclose(i, j)
        for i, j in zip(c, wc.waterH1):
            npt.assert_allclose(i, j)
        for i, j in zip(d, wc.waterH2):
            npt.assert_allclose(i, j)


def test_save_results_onlyO():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        wc = WaterClustering(10)
        wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
        wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
        wc._water_type.append("O_clust")
        wc._water_type.append("O_clust")
        wc.save_results(f.name)
        a, b, _, _ = read_results(f.name)
        for i, j in zip(a, wc.water_type):
            assert i == j
        for i, j in zip(b, wc.waterO):
            npt.assert_allclose(i, j)


def test_delete_data_onlyO():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as dat:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res:
            wc = WaterClustering(10, output_file=res, restart_data_file=dat)
            Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
            Onew, _, _ = wc._delete_data([0, 2], Odata=Odata)
            npt.assert_allclose(Onew[0], Odata[1])
            assert os.path.isfile(dat.name)


def test_delete_data_all_waters():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as dat:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res:
            wc = WaterClustering(10, output_file=res, restart_data_file=dat)
            Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
            H1 = np.asarray([[0.8, 0.5, 0.8], [1.4, 0.6, 3.7], [3.9, 5.1, 5.9]])
            H2 = np.asarray([[0.4, 0.7, 0.1], [1.7, 3.6, 2.7], [1.8, 3.8, 5.1]])
            Onew, H1new, H2new = wc._delete_data([1, 2], Odata, H1, H2)
            npt.assert_allclose(Onew[0], Odata[0])
            npt.assert_allclose(H1new[0], H1[0])
            npt.assert_allclose(H2new[0], H2[0])
            assert os.path.isfile(dat.name)


def test_delete_data_not_restart():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as dat:
        wc = WaterClustering(10, restart_data_file=dat)
        Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
        H1 = np.asarray([[0.8, 0.5, 0.8], [1.4, 0.6, 3.7], [3.9, 5.1, 5.9]])
        H2 = np.asarray([[0.4, 0.7, 0.1], [1.7, 3.6, 2.7], [1.8, 3.8, 5.1]])
        _, _, _ = wc._delete_data([1, 2], Odata, H1, H2)
        assert os.path.isfile(dat.name)


def test_save_clustering_options():
    ca = "OPTICS"
    whichH = ["onlyO"]
    wc = WaterClustering(
        10,
        clustering_algorithm=ca,
        water_types_to_find=whichH,
        restart_after_found=True,
    )
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        wc._save_clustering_options(fname=f.name)
        with open(f.name, "r") as f2:
            lines = f2.readlines()
            assert int(lines[0]) == wc.nsnaps
            assert lines[1].strip() == wc.clustering_algorithm
            assert lines[2].strip() == " ".join(wc.water_types_to_find)
            assert (lines[3].strip() == "True") == wc.restart_after_find
            assert np.allclose([float(x) for x in lines[4].split()], wc.min_samples)
            assert np.allclose([float(x) for x in lines[5].split()], wc.xis)
            assert float(lines[6]) == wc.numbpct_oxygen
            assert bool(lines[7].strip()) == wc.normalize_orientations
            assert float(lines[8]) == wc.numbpct_hyd_orient_analysis
            assert float(lines[9]) == wc.kmeans_ang_cutoff
            assert float(lines[10]) == wc.kmeans_inertia_cutoff
            assert float(lines[11]) == wc.conserved_angdiff_cutoff
            assert float(lines[12]) == wc.conserved_angstd_cutoff
            assert float(lines[13]) == wc.other_waters_hyd_minsamp_pct
            assert float(lines[14]) == wc.noncon_angdiff_cutoff
            assert float(lines[15]) == wc.halfcon_angstd_cutoff
            assert float(lines[16]) == wc.weakly_angstd_cutoff
            assert float(lines[17]) == wc.weakly_explained
            assert np.allclose([float(x) for x in lines[18].split()], wc.xiFCW)
            assert np.allclose([float(x) for x in lines[19].split()], wc.xiHCW)
            assert np.allclose([float(x) for x in lines[20].split()], wc.xiWCW)
            assert int(lines[21]) == wc.njobs
            assert int(lines[22].strip()) == wc.verbose
            assert int(lines[23].strip()) == wc.debugO
            assert int(lines[24].strip()) == wc.debugH
            assert (lines[25].strip() == "True") == wc.plotreach
            assert (lines[26].strip() == "True") == wc.plotend


def test_create_from_file():
    ca = "OPTICS"
    whichH = ["onlyO"]
    wc = WaterClustering(10, clustering_algorithm=ca, water_types_to_find=whichH)
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        wc._save_clustering_options(f.name)
        newWC = WaterClustering.create_from_file(f.name)
        d1 = wc.__dict__
        d2 = newWC.__dict__
        for i in d1.keys():
            assert d1[i] == d2[i]


def test_read_and_set_water_clust_options_file_not_found():
    wc = WaterClustering(9)
    try:
        wc.read_and_set_water_clust_options("tests/data/nonexistent.dat")
    except Exception as e:
        assert str(e) == "output file not found"


def test_restart_cluster_onlyO():
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as partial_data_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as partial_results_file:
        # create partial data file
        Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
        np.savetxt(partial_data_file.name, Odata)
        # create partial results file
        wc = WaterClustering(10, water_types_to_find=["onlyO"])
        wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
        wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
        wc._water_type.append("O_clust")
        wc._water_type.append("O_clust")
        wc.save_results(partial_results_file.name)
        # restart clustering
        wc.restart_cluster(partial_results_file.name, partial_data_file.name)
        # check results
        assert wc.water_type == ["O_clust", "O_clust"]
        npt.assert_allclose(wc.waterO, np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))


def test_restart_cluster_water_types():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as res:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as dat:
            ca = "OPTICS"
            whichH = ["FCW", "HCW", "WCW"]
            wc = WaterClustering(
                10, clustering_algorithm=ca, water_types_to_find=whichH
            )
            wc._waterO.append(np.asarray([0.0, 0.0, 0.0]))
            wc._waterO.append(np.asarray([0.0, 2.0, 0.0]))
            wc._waterH1.append(np.asarray([1.0, 0.0, 0.0]))
            wc._waterH1.append(np.asarray([0.0, 0.8, 0.5]))
            wc._waterH2.append(np.asarray([2.5, 2.0, 2.8]))
            wc._waterH2.append(np.asarray([0.0, 3.0, 0.0]))
            wc._water_type.append("FCW")
            wc._water_type.append("HCW")
            wc.save_results(res.name)
            Odata = np.asarray(
                [
                    [0.1, 0.1, 0.1, 1, 1, 0, -1, 1, 0],
                    [1.5, 1.6, 1.7, 2.5, 2.5, 0, 0, 2.5, 2.5],
                    [1.9, 5.8, 5.6, 1, 5, 5, 1, 5, 6.5],
                ]
            )
            np.savetxt(dat.name, np.c_[Odata])
            newWC = WaterClustering(10)
            newWC.restart_cluster(res.name, dat.name)


def test_restart_cluster_and_create_class_from_file():
    WaterClustering.create_from_files_and_restart(
        "tests/data/restart_partial_results.dat", "tests/data/restart_data.dat"
    )


def test_single_clustering_OPTICS():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.single_clustering(*get_orientations_from_positions(Opos, Hpos))
    a, b, c, d = read_results("tests/data/Single_OPTICS.dat")
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_single_clustering_HDBSCAN():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.single_clustering(
        *get_orientations_from_positions(Opos, Hpos), clustering_algorithm="HDBSCAN"
    )
    a, b, c, d = read_results("tests/data/Single_HDBSCAN.dat")
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))
    # sort wc.waterO according to b
    # b is xyz coordinates of water Oxygen. Sort them according to x
    b = np.asarray(b)
    b = b[b[:, 0].argsort()]
    c = np.asarray(c)
    c = c[c[:, 0].argsort()]
    d = np.asarray(d)
    d = d[d[:, 0].argsort()]
    # now dothe same for water
    b1 = np.asarray(wc.waterO)
    b1 = b1[b1[:, 0].argsort()]
    c1 = np.asarray(wc.waterH1)
    c1 = c1[c1[:, 0].argsort()]
    d1 = np.asarray(wc.waterH2)
    d1 = d1[d1[:, 0].argsort()]
    npt.assert_allclose(b, b1)
    npt.assert_allclose(c, c1)
    npt.assert_allclose(d, d1)


def test_multistage_reclustering_OPTICS():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.multi_stage_reclustering(*get_orientations_from_positions(Opos, Hpos))
    a, b, c, d = read_results("tests/data/MSR_OPTICS.dat")
    npt.assert_allclose(b, wc.waterO)
    npt.assert_allclose(c, wc.waterH1)
    npt.assert_allclose(d, wc.waterH2)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_multistage_reclustering_HDBSCAN():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.multi_stage_reclustering(
        *get_orientations_from_positions(Opos, Hpos), clustering_algorithm="HDBSCAN"
    )
    a, b, c, d = read_results("tests/data/MSR_HDBSCAN.dat")
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))
    tol = 1e-1
    # b is xyz coordinates of water Oxygen. Sort them according to x
    b = np.asarray(b)
    b = b[b[:, 0].argsort()]
    c = np.asarray(c)
    c = c[c[:, 0].argsort()]
    d = np.asarray(d)
    d = d[d[:, 0].argsort()]
    # now dothe same for water
    b1 = np.asarray(wc.waterO)
    b1 = b1[b1[:, 0].argsort()]
    c1 = np.asarray(wc.waterH1)
    c1 = c1[c1[:, 0].argsort()]
    d1 = np.asarray(wc.waterH2)
    d1 = d1[d1[:, 0].argsort()]
    npt.assert_allclose(b, b1, atol=tol)
    npt.assert_allclose(c, c1, atol=tol)
    npt.assert_allclose(d, d1, atol=tol)


def test_quick_multistage_reclustering_OPTICS():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.quick_multi_stage_reclustering(*get_orientations_from_positions(Opos, Hpos))
    a, b, c, d = read_results("tests/data/MSR_OPTICS.dat")
    tol = 1e-4
    # b is xyz coordinates of water Oxygen. Sort them according to x
    b = np.asarray(b)
    b = b[b[:, 0].argsort()]
    c = np.asarray(c)
    c = c[c[:, 0].argsort()]
    d = np.asarray(d)
    d = d[d[:, 0].argsort()]
    # now do the same for water
    b1 = np.asarray(wc.waterO)
    b1 = b1[b1[:, 0].argsort()]
    c1 = np.asarray(wc.waterH1)
    c1 = c1[c1[:, 0].argsort()]
    d1 = np.asarray(wc.waterH2)
    d1 = d1[d1[:, 0].argsort()]
    npt.assert_allclose(b, b1, atol=tol)
    npt.assert_allclose(c, c1, atol=tol)
    npt.assert_allclose(d, d1, atol=tol)
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))


def test_quick_multistage_reclustering_HDBSCAN():
    Nsnap = 20
    Opos = np.loadtxt("tests/data/testdataO.dat")
    Hpos = np.loadtxt("tests/data/testdataH.dat")
    wc = WaterClustering(Nsnap)
    wc.quick_multi_stage_reclustering(
        *get_orientations_from_positions(Opos, Hpos), clustering_algorithm="HDBSCAN"
    )
    a, b, c, d = read_results("tests/data/MSR_HDBSCAN.dat")
    assert all(wc.water_type[i] == a[i] for i in range(len(wc.water_type)))
    tol = 1e-1
    # b is xyz coordinates of water Oxygen. Sort them according to x
    b = np.asarray(b)
    b = b[b[:, 0].argsort()]
    c = np.asarray(c)
    c = c[c[:, 0].argsort()]
    d = np.asarray(d)
    d = d[d[:, 0].argsort()]
    # now do the same for water
    b1 = np.asarray(wc.waterO)
    b1 = b1[b1[:, 0].argsort()]
    c1 = np.asarray(wc.waterH1)
    c1 = c1[c1[:, 0].argsort()]
    d1 = np.asarray(wc.waterH2)
    d1 = d1[d1[:, 0].argsort()]
    npt.assert_allclose(b, b1, atol=tol)
    npt.assert_allclose(c, c1, atol=tol)
    npt.assert_allclose(d, d1, atol=tol)
