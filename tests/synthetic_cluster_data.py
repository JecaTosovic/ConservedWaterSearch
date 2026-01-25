import numpy as np

from ConservedWaterSearch.hydrogen_orientation import hydrogen_orientation_analysis

WATER_ANGLE_DEG = 104.5
N_SNAPSHOTS = 30
N_WATERS = 6
BOND_LENGTH = 1.0
OXYGEN_SIGMA = 0.05
H_FCW_SIGMA = 0.01
H_HCW_SIGMA = 0.02
H_WCW_SIGMA = 0.02


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    return vec / np.linalg.norm(vec)


def _random_perpendicular(
    rng: np.random.Generator, u: np.ndarray
) -> np.ndarray:
    vec = rng.normal(size=3)
    vec = vec - np.dot(vec, u) * u
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return _random_perpendicular(rng, u)
    return vec / norm


def _orthonormal_basis(
    rng: np.random.Generator, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v = _random_perpendicular(rng, u)
    w = np.cross(u, v)
    return v, w / np.linalg.norm(w)


def _make_h2_direction(
    rng: np.random.Generator, h1: np.ndarray
) -> np.ndarray:
    theta = np.deg2rad(WATER_ANGLE_DEG)
    perp = _random_perpendicular(rng, h1)
    return np.cos(theta) * h1 + np.sin(theta) * perp


def _noisy_unit(
    rng: np.random.Generator, vec: np.ndarray, sigma: float
) -> np.ndarray:
    noisy = vec + rng.normal(scale=sigma, size=3)
    return noisy / np.linalg.norm(noisy)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    dot = max(-1.0, min(1.0, dot))
    return float(np.rad2deg(np.arccos(dot)))


def _make_wcw_pair(
    rng: np.random.Generator,
    existing_h1: list[np.ndarray],
    existing_h2: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    min_sep_deg = 70.0
    for _ in range(400):
        h1b = _random_unit(rng)
        h2b = _make_h2_direction(rng, h1b)
        if any(_angle_deg(h1b, h1) <= min_sep_deg for h1 in existing_h1):
            continue
        if any(_angle_deg(h2b, h2) <= min_sep_deg for h2 in existing_h2):
            continue
        cross_angles = [
            _angle_deg(h1b, h2) for h2 in existing_h2
        ] + [
            _angle_deg(h2b, h1) for h1 in existing_h1
        ]
        if all(abs(ang - WATER_ANGLE_DEG) > 20.0 for ang in cross_angles):
            return h1b, h2b
    raise RuntimeError("Failed to generate WCW orientations with separation.")


def _build_centers(layout: str) -> np.ndarray:
    centers = []
    if layout == "xy":
        for idx in range(N_WATERS):
            centers.append([8.0 * (idx % 3), 8.0 * (idx // 3), 0.0])
    elif layout == "yz":
        for idx in range(N_WATERS):
            centers.append([24.0, 9.0 * (idx % 3), 9.0 * (idx // 3)])
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return np.asarray(centers)


def generate_dataset(
    seed: int, layout: str, hcw_modes: list[str] | None = None
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    centers = _build_centers(layout)
    water_types = ["FCW"] * 1 + ["HCW"] * 2 + ["WCW"] * 3
    if hcw_modes is None:
        hcw_modes = ["bimodal", "random"]
    if len(hcw_modes) != 2:
        raise ValueError("hcw_modes must contain two entries.")

    expected_orient_centers = []
    expected_hcw_modes = []
    base_orientations = []
    hcw_idx = 0

    for wtype in water_types:
        if wtype == "FCW":
            h1 = _random_unit(rng)
            h2 = _make_h2_direction(rng, h1)
            base_orientations.append(("FCW", h1, h2))
            expected_orient_centers.append([h1, h2])
            expected_hcw_modes.append(None)
        elif wtype == "HCW":
            mode = hcw_modes[hcw_idx]
            hcw_idx += 1
            for _ in range(200):
                h1 = _random_unit(rng)
                v, w = _orthonormal_basis(rng, h1)
                theta = np.deg2rad(WATER_ANGLE_DEG)
                if mode == "bimodal":
                    phi = np.deg2rad(rng.uniform(100.0, 175.0))
                    h2_a = np.cos(theta) * h1 + np.sin(theta) * v
                    h2_b = np.cos(theta) * h1 + np.sin(theta) * (
                        np.cos(phi) * v + np.sin(phi) * w
                    )
                    h1_seq = []
                    h2_seq = []
                    for snap in range(N_SNAPSHOTS):
                        h1_seq.append(_noisy_unit(rng, h1, H_HCW_SIGMA))
                        base = h2_a if snap % 2 == 0 else h2_b
                        h2_seq.append(_noisy_unit(rng, base, H_HCW_SIGMA))
                    orientations = np.vstack([h1_seq, h2_seq])
                    res = hydrogen_orientation_analysis(orientations)
                    if res and res[0][2] == "HCW":
                        base_orientations.append(
                            ("HCW", mode, np.asarray(h1_seq), np.asarray(h2_seq))
                        )
                        expected_orient_centers.append([h1, h2_a, h2_b])
                        expected_hcw_modes.append(mode)
                        break
                elif mode == "random":
                    h1_seq = []
                    h2_seq = []
                    for _ in range(N_SNAPSHOTS):
                        h1_seq.append(_noisy_unit(rng, h1, H_HCW_SIGMA))
                        h2_base = _make_h2_direction(rng, h1)
                        h2_seq.append(_noisy_unit(rng, h2_base, H_HCW_SIGMA))
                    orientations = np.vstack([h1_seq, h2_seq])
                    res = hydrogen_orientation_analysis(orientations)
                    if res and res[0][2] == "HCW":
                        base_orientations.append(
                            ("HCW", mode, np.asarray(h1_seq), np.asarray(h2_seq))
                        )
                        expected_orient_centers.append([h1])
                        expected_hcw_modes.append(mode)
                        break
                else:
                    raise ValueError(f"Unknown HCW mode: {mode}")
            else:
                raise RuntimeError(f"Failed to generate HCW orientations for {mode}.")
        elif wtype == "WCW":
            for _ in range(200):
                h1a = _random_unit(rng)
                h2a = _make_h2_direction(rng, h1a)
                h1b, h2b = _make_wcw_pair(rng, [h1a], [h2a])
                h1c, h2c = _make_wcw_pair(rng, [h1a, h1b], [h2a, h2b])
                h1_seq = []
                h2_seq = []
                for snap in range(N_SNAPSHOTS):
                    if snap < 12:
                        pair_idx = 0
                    elif snap < 24:
                        pair_idx = 1
                    else:
                        pair_idx = 2
                    h1_base = [h1a, h1b, h1c][pair_idx]
                    h2_base = [h2a, h2b, h2c][pair_idx]
                    h1_seq.append(_noisy_unit(rng, h1_base, H_WCW_SIGMA))
                    h2_seq.append(_noisy_unit(rng, h2_base, H_WCW_SIGMA))
                orientations = np.vstack([h1_seq, h2_seq])
                res = hydrogen_orientation_analysis(orientations)
                if res and res[0][2] == "WCW":
                    base_orientations.append(
                        ("WCW", np.asarray(h1_seq), np.asarray(h2_seq))
                    )
                    expected_orient_centers.append(
                        [h1a, h2a, h1b, h2b, h1c, h2c]
                    )
                    expected_hcw_modes.append(None)
                    break
            else:
                raise RuntimeError("Failed to generate WCW orientations.")
        else:
            raise ValueError(f"Unknown water type: {wtype}")

    rows = []
    for snap in range(N_SNAPSHOTS):
        for idx, wtype in enumerate(water_types):
            center = centers[idx]
            if wtype == "HCW":
                oxygen = center.copy()
            else:
                oxygen = center + rng.normal(scale=OXYGEN_SIGMA, size=3)
            orient = base_orientations[idx]
            if wtype == "FCW":
                h1_dir = _noisy_unit(rng, orient[1], H_FCW_SIGMA)
                h2_dir = _noisy_unit(rng, orient[2], H_FCW_SIGMA)
            elif wtype == "HCW":
                h1_dir = orient[2][snap]
                h2_dir = orient[3][snap]
            else:
                h1_dir = orient[1][snap]
                h2_dir = orient[2][snap]
            h1 = oxygen + BOND_LENGTH * h1_dir
            h2 = oxygen + BOND_LENGTH * h2_dir
            rows.append(np.concatenate([oxygen, h1, h2]))

    expected = {
        "centers": centers,
        "types": water_types,
        "orient_centers": expected_orient_centers,
        "hcw_modes": expected_hcw_modes,
    }
    return np.asarray(rows), expected


DATASET_CONFIGS = {
    "set1": {"seed": 222, "layout": "yz", "hcw_modes": ["random", "bimodal"]},
    "set2": {"seed": 444, "layout": "yz", "hcw_modes": ["random", "bimodal"]},
}


def dataset_path(name: str) -> str:
    return f"tests/data/synthetic_cluster_{name}.dat"


def expected_for(name: str) -> dict:
    config = DATASET_CONFIGS[name]
    _, expected = generate_dataset(
        config["seed"], config["layout"], config["hcw_modes"]
    )
    return expected


def write_pdb(path: str, data: np.ndarray) -> None:
    lines = []
    atom_idx = 1
    res_idx = 1
    for row_idx, row in enumerate(data):
        if row_idx % N_WATERS == 0 and row_idx > 0:
            lines.append("TER\n")
        coords = row.reshape(3, 3)
        for atom_name, element, xyz in zip(
            ("O", "H1", "H2"),
            ("O", "H", "H"),
            coords,
        ):
            lines.append(
                "ATOM  {atom:5d} {name:<4s} HOH A{res:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el:>2s}\n".format(
                    atom=atom_idx,
                    name=atom_name,
                    res=res_idx,
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                    el=element,
                )
            )
            atom_idx += 1
        res_idx += 1
    lines.append("END\n")
    with open(path, "w") as handle:
        handle.writelines(lines)


def write_pdb_for_configs() -> None:
    for name, config in DATASET_CONFIGS.items():
        data, _ = generate_dataset(
            config["seed"], config["layout"], config["hcw_modes"]
        )
        write_pdb(f"tests/data/synthetic_cluster_{name}.pdb", data)
