import tempfile

import numpy as np
import pytest

from ConservedWaterSearch.water_clustering import WaterClustering

WATER_ANGLE_DEG = 104.5


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    return vec / np.linalg.norm(vec)


def _random_perpendicular(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    vec = vec - np.dot(vec, u) * u
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return _random_perpendicular(u, rng)
    return vec / norm


def _make_h2_direction(h1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = np.deg2rad(WATER_ANGLE_DEG)
    perp = _random_perpendicular(h1, rng)
    return np.cos(theta) * h1 + np.sin(theta) * perp


def _noisy_unit(vec: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noisy = vec + rng.normal(scale=sigma, size=3)
    return noisy / np.linalg.norm(noisy)


def generate_orientation_sample(
    kind: str, nsnaps: int, rng: np.random.Generator
) -> np.ndarray:
    base_h1 = _random_unit_vector(rng)
    base_h2 = _make_h2_direction(base_h1, rng)
    orientations = []
    if kind == "FCW":
        for _ in range(nsnaps):
            orientations.append(_noisy_unit(base_h1, rng, 0.02))
            orientations.append(_noisy_unit(base_h2, rng, 0.02))
    elif kind == "HCW":
        for _ in range(nsnaps):
            orientations.append(base_h1)
            h2 = _make_h2_direction(base_h1, rng)
            orientations.append(_noisy_unit(h2, rng, 0.01))
    elif kind == "WCW":
        base_a = _random_unit_vector(rng)
        base_b = _make_h2_direction(base_a, rng)
        base_c = _random_perpendicular(base_a, rng)
        base_d = _make_h2_direction(base_c, rng)
        clusters = [base_a, base_b, base_c, base_d]
        cluster_size = (2 * nsnaps) // len(clusters)
        for base in clusters:
            for _ in range(cluster_size):
                orientations.append(_noisy_unit(base, rng, 0.03))
    elif kind == "not_conserved":
        for _ in range(nsnaps):
            h1 = _noisy_unit(base_h1, rng, 0.01)
            h2 = _noisy_unit(base_h1, rng, 0.01)
            orientations.append(h1)
            orientations.append(h2)
    else:
        msg = f"Unknown kind: {kind}"
        raise ValueError(msg)
    return np.asarray(orientations)


def generate_water_network(
    nsnaps: int,
    n_waters: int,
    rng: np.random.Generator,
    oxygen_sigma: float = 0.05,
    hydrogen_sigma: float = 0.02,
    bond_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_oxygens = np.stack([np.array([4.0 * i, 0.0, 0.0]) for i in range(n_waters)])
    base_orientations = []
    for _ in range(n_waters):
        h1 = _random_unit_vector(rng)
        h2 = _make_h2_direction(h1, rng)
        base_orientations.append((h1, h2))
    Opos = []
    Hpos = []
    for _ in range(nsnaps):
        for water_idx in range(n_waters):
            base_O = base_oxygens[water_idx]
            h1_dir, h2_dir = base_orientations[water_idx]
            oxygen = base_O + rng.normal(scale=oxygen_sigma, size=3)
            h1 = oxygen + bond_length * _noisy_unit(h1_dir, rng, hydrogen_sigma)
            h2 = oxygen + bond_length * _noisy_unit(h2_dir, rng, hydrogen_sigma)
            Opos.append(oxygen)
            Hpos.append(h1)
            Hpos.append(h2)
    return np.asarray(Opos), np.asarray(Hpos), base_oxygens


@pytest.fixture()
def orientations_normalized():
    return np.asarray([[1, 0, 0], [-0.25038 * 2, 0.96814764 * 2, 0]])


@pytest.fixture()
def orientations_not_normalized():
    return np.asarray([[1, 0, 0], [-0.25038, 0.96814764, 0]])


@pytest.fixture(
    params=[
        ("FCW", "FCW"),
        ("HCW", "HCW"),
        ("WCW", "WCW"),
        ("not_conserved", None),
    ]
)
def water_data(request):
    rng = np.random.default_rng(1234)
    kind, expected = request.param
    return generate_orientation_sample(kind, nsnaps=24, rng=rng), expected


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
    rng = np.random.default_rng(2024)
    Opos, Hpos, centers = generate_water_network(nsnaps=12, n_waters=2, rng=rng)
    return Opos, Hpos, centers
