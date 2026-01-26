import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

from ConservedWaterSearch.utils import (
    get_orientations_from_positions,
    read_results,
)
from ConservedWaterSearch.water_clustering import WaterClustering
from tests.synthetic_cluster_data import (
    DATASET_CONFIGS,
    N_SNAPSHOTS,
    N_WATERS,
    WATER_ANGLE_DEG,
    expected_for,
    generate_dataset,
)

np_major = int(np.__version__.split(".")[0])


def test_save_results(water_clustering_setup):
    wc, onlyO = water_clustering_setup
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        wc.save_results(f.name)
        a, b, c, d = read_results(f.name)

        # Assertions for both scenarios
        for i, j in zip(a, wc.water_type):
            assert i == j
        for i, j in zip(b, wc.waterO):
            npt.assert_allclose(i, j)

        if not onlyO:
            # Additional assertions for hydrogen presence
            for i, j in zip(c, wc.waterH1):
                npt.assert_allclose(i, j)
            for i, j in zip(d, wc.waterH2):
                npt.assert_allclose(i, j)


def test_delete_data_not_restart():
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as dat:
        wc = WaterClustering(10, restart_data_file=dat)
        Odata = np.asarray([[0.1, 0.1, 0.1], [1.5, 1.6, 1.7], [1.9, 5.8, 5.6]])
        H1 = np.asarray([[0.8, 0.5, 0.8], [1.4, 0.6, 3.7], [3.9, 5.1, 5.9]])
        H2 = np.asarray([[0.4, 0.7, 0.1], [1.7, 3.6, 2.7], [1.8, 3.8, 5.1]])
        _, _, _ = wc._delete_data([1, 2], Odata, H1, H2)
        assert os.path.isfile(dat.name)


def test_delete_data_onlyO(water_clustering_setup_for_deletion):
    wc, dat_name, Odata, _, _ = water_clustering_setup_for_deletion
    Onew, _, _ = wc._delete_data([0, 2], Odata=Odata)
    npt.assert_allclose(Onew, np.asarray([Odata[1]]))
    assert os.path.isfile(dat_name)


def test_delete_data_all_waters(water_clustering_setup_for_deletion):
    wc, dat_name, Odata, H1, H2 = water_clustering_setup_for_deletion
    Onew, H1new, H2new = wc._delete_data([1, 2], Odata, H1, H2)
    npt.assert_allclose(Onew, np.asarray([Odata[0]]))
    npt.assert_allclose(H1new, np.asarray([H1[0]]))
    npt.assert_allclose(H2new, np.asarray([H2[0]]))
    assert os.path.isfile(dat_name)


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
        with open(f.name) as f2:
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
    whichH = "onlyO"
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
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = os.path.join(tmpdir, "nonexistent.dat")
        with pytest.raises(FileNotFoundError, match="output file not found"):
            wc.read_and_set_water_clust_options(missing)


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


def test_restart_cluster_and_create_class_from_file(water_clustering_data):
    Opos, Hpos, _ = water_clustering_data
    Odata, H1, H2 = get_orientations_from_positions(Opos, Hpos)
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as partial_data, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as partial_results:
        np.savetxt(partial_data.name, np.c_[Odata, H1, H2])
        wc = WaterClustering(12)
        wc.single_clustering(Odata, H1, H2)
        wc.save_results(partial_results.name)
        WaterClustering.create_from_files_and_restart(
            partial_results.name, partial_data.name
        )


def sort_data_by_x(data):
    data = np.asarray(data)
    return data[data[:, 0].argsort()]


def _load_cluster_dataset(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    data, _ = generate_dataset(config["seed"], config["layout"], config["hcw_modes"])
    Opos = data[:, :3]
    H1pos = data[:, 3:6]
    H2pos = data[:, 6:9]
    coordsH = np.empty((len(Opos) * 2, 3))
    coordsH[0::2] = H1pos
    coordsH[1::2] = H2pos
    return get_orientations_from_positions(Opos, coordsH)


def _match_results_to_centers(wc, centers, tol=0.25):
    mapping = {i: [] for i in range(len(centers))}
    for idx, Opos in enumerate(wc.waterO):
        dists = np.linalg.norm(centers - Opos, axis=1)
        best = int(np.argmin(dists))
        assert dists[best] < tol
        mapping[best].append(idx)
    return mapping


def _expanded_centers(centers):
    normed = []
    for center_vec in centers:
        normed_center = center_vec / np.linalg.norm(center_vec)
        normed.append(normed_center)
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            summed = normed[i] + normed[j]
            norm = np.linalg.norm(summed)
            if norm > 1e-6:
                normed.append(summed / norm)
    return normed


def _orientation_close(vec, centers, tol=0.25):
    vec = vec / np.linalg.norm(vec)
    for center_vec in _expanded_centers(centers):
        center_normed = center_vec / np.linalg.norm(center_vec)
        if np.linalg.norm(vec - center_normed) < tol:
            return True
    return False


def _angle_near_water(h1, h2, tol_deg=15.0):
    h1 = h1 / np.linalg.norm(h1)
    h2 = h2 / np.linalg.norm(h2)
    dot = float(np.dot(h1, h2))
    dot = max(-1.0, min(1.0, dot))
    angle = np.rad2deg(np.arccos(dot))
    return abs(angle - 104.5) < tol_deg


@pytest.mark.parametrize("dataset_name", sorted(DATASET_CONFIGS.keys()))
def test_onlyO_mode_clustering(dataset_name):
    Odata, _, _ = _load_cluster_dataset(dataset_name)
    wc = WaterClustering(30, water_types_to_find=["onlyO"])
    wc.single_clustering(
        Odata, None, None, whichH=["onlyO"], clustering_algorithm="OPTICS"
    )
    assert len(wc.waterO) == 6
    assert all(wt == "O_clust" for wt in wc.water_type)


@pytest.mark.parametrize("dataset_name", sorted(DATASET_CONFIGS.keys()))
@pytest.mark.parametrize(
    ("clustering_func", "algorithm"),
    [
        ("single_clustering", "OPTICS"),
        ("single_clustering", "HDBSCAN"),
        ("multi_stage_reclustering", "OPTICS"),
        ("multi_stage_reclustering", "HDBSCAN"),
        ("quick_multi_stage_reclustering", "OPTICS"),
        ("quick_multi_stage_reclustering", "HDBSCAN"),
    ],
)
def test_clustering_all_water_types(dataset_name, clustering_func, algorithm):
    Odata, H1, H2 = _load_cluster_dataset(dataset_name)
    expected = expected_for(dataset_name)
    wc = WaterClustering(30)
    func = getattr(wc, clustering_func)
    func(Odata, H1, H2, clustering_algorithm=algorithm)

    mapping = _match_results_to_centers(wc, expected["centers"])
    assert len(mapping) == 6

    for water_idx, result_indices in mapping.items():
        assert result_indices
        expected_type = expected["types"][water_idx]
        hcw_mode = expected["hcw_modes"][water_idx]
        for res_idx in result_indices:
            assert wc.water_type[res_idx] == expected_type
            h1_vec = wc.waterH1[res_idx] - wc.waterO[res_idx]
            h2_vec = wc.waterH2[res_idx] - wc.waterO[res_idx]
            centers = expected["orient_centers"][water_idx]
            assert _orientation_close(h1_vec, centers)
            if expected_type == "HCW" and hcw_mode == "random":
                pass
            else:
                assert _orientation_close(h2_vec, centers)
            if expected_type == "FCW" or (
                expected_type == "HCW" and hcw_mode != "random"
            ):
                assert _angle_near_water(h1_vec, h2_vec)


def _angle_deg(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    dot = float(np.dot(vec1, vec2))
    dot = max(-1.0, min(1.0, dot))
    return float(np.rad2deg(np.arccos(dot)))


@pytest.mark.parametrize("dataset_name", sorted(DATASET_CONFIGS.keys()))
def test_hcw_generation_modes(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    data, expected = generate_dataset(
        config["seed"], config["layout"], config["hcw_modes"]
    )
    per_water = [data[i::N_WATERS] for i in range(N_WATERS)]
    for idx, wtype in enumerate(expected["types"]):
        if wtype != "HCW":
            continue
        mode = expected["hcw_modes"][idx]
        water = per_water[idx]
        Opos = water[:, :3]
        H1pos = water[:, 3:6]
        H2pos = water[:, 6:9]
        h1_dirs = H1pos - Opos
        h2_dirs = H2pos - Opos
        h1_dirs /= np.linalg.norm(h1_dirs, axis=1, keepdims=True)
        h2_dirs /= np.linalg.norm(h2_dirs, axis=1, keepdims=True)
        angles = np.array([_angle_deg(h1, h2) for h1, h2 in zip(h1_dirs, h2_dirs)])
        assert abs(float(np.mean(angles)) - WATER_ANGLE_DEG) < 5.0
        if mode == "bimodal":
            centers = expected["orient_centers"][idx][1:]
            assert len(centers) == 2
            center_angle = _angle_deg(centers[0], centers[1])
            assert center_angle >= 60.0
            counts = [0, 0]
            for h2 in h2_dirs:
                d0 = _angle_deg(h2, centers[0])
                d1 = _angle_deg(h2, centers[1])
                counts[int(d1 < d0)] += 1
            assert min(counts) >= int(0.3 * N_SNAPSHOTS)
        elif mode == "random":
            h2_mean = np.mean(h2_dirs, axis=0)
            assert np.linalg.norm(h2_mean) < 0.7
        else:
            msg = f"Unexpected HCW mode: {mode}"
            raise AssertionError(msg)
