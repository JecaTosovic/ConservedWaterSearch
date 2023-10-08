from __future__ import annotations
import os
import platform
from re import L
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from nglview import NGLWidget
    except ImportError:
        NGLWidget = None
    try:
        import pymol
        from pymol import cmd
    except ImportError:
        pymol = None
        cmd = None


def __check_mpl_installation():
    """Check if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise Exception(
            "install matplotlib using conda install -c conda-forge matplotlib or pip install matplotlib"
        )
    return plt


def read_results(
    fname: str = "Clustering_results.dat",
    typefname: str = "Type_Clustering_results.dat",
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read results from files.

    Read results from files and return them in order for further
    processing.

    Args:
        fname (str, optional): File name of the file that contains
            water coordinates. Defaults to "Clustering_results.dat".
        typefname (str, optional): File name of the file that contains
            water classification strings.
            Defaults to "Type_Clustering_results.dat".

    Returns:
        tuple[ list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray] ]:
        returns list of strings which represents water types, and arrays
        of locations of oxygen and two hyrogens. If only oxygens were
        saved returned hydrogen coordinates are empty arrays

    Examples::

        water_types, coord_O, coord_H1, coord_H2 = read_results(
            fname = "Clust_res.dat",
            typefname = "Type_Clust_res.dat",
        )
    """
    water_type = []
    waterO = []
    waterH1 = []
    waterH2 = []
    coords = np.loadtxt(fname)
    if coords.shape[1] == 3:
        for i in coords:
            waterO.append(i)
    else:
        Opos = coords[:, :3]
        H1 = coords[:, 3:6]
        H2 = coords[:, 6:9]
        for i, j, k in zip(Opos, H1, H2):
            waterO.append(i)
            waterH1.append(j)
            waterH2.append(k)
    types = np.loadtxt(typefname, dtype=str)
    for i in types:
        water_type.append(i)
    return (
        water_type,
        list(np.asarray(waterO)),
        list(np.asarray(waterH1)),
        list(np.asarray(waterH2)),
    )


def get_orientations_from_positions(
    coordsO: np.ndarray, coordsH: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns orientations from coordinates.

    Calculates relative orientations of hydrogen atoms from their
    positions. The output of this function can be used as input for
    water clustering.

    Args:
        coordsO (np.ndarray): Oxygen coordinates - shape
            (N_waters, 3)
        coordsH (np.ndarray): Hydrogen coordinates - two
            hydrogens bound to the same oxygen have to be placed
            one after another in the array. Shape: (2*N_waters, 3).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        returns oxygen coordinates array and two hydrogen orientation
        arrays.
    """
    Odata: np.ndarray = np.asarray(coordsO)
    if len(coordsH) > 1:
        H1: np.ndarray = coordsH[::2, :]
        H2: np.ndarray = coordsH[1:, :]
        H2 = H2[::2, :]
        v1: list[np.ndarray] = []
        v2: list[np.ndarray] = []
        for o, h1, h2 in zip(Odata, H1, H2):
            a: np.ndarray = h1 - o
            b: np.ndarray = h2 - o
            if (np.linalg.norm(h1 - o) > 1.2) or (np.linalg.norm(h2 - o) > 1.2):
                raise ValueError(
                    "bad input HO bonds in water longer than 1.2A; value:",
                    np.linalg.norm(h1 - o),
                    np.linalg.norm(h2 - o),
                )
            v1.append(a)
            v2.append(b)
        H1orientdata: np.ndarray = np.asarray(v1)
        H2orientdata: np.ndarray = np.asarray(v2)
        return Odata, H1orientdata, H2orientdata
    else:
        raise Exception("Hydrogen array of wrong length")


def _make_protein_surface_with_ligand():
    from pymol import cmd

    protein = cmd.get_unused_name("only_protein_")
    cmd.select(protein, "polymer")
    povrsina = cmd.get_unused_name("protein_surface_")
    cmd.create(povrsina, protein)
    cmd.show("surface", povrsina)
    cmd.color("gray70", povrsina)
    cmd.set("transparency", 0.5, povrsina)
    # ligand representation
    ligand = cmd.get_unused_name("ligand_")
    cmd.select(ligand, "organic")
    cmd.show("licorice", ligand)


def _add_polar_contacts(waters: str, aminokis_u_am: str | None = None):
    from pymol import cmd

    if aminokis_u_am is not None:
        sele = aminokis_u_am + " or " + waters + " or organic"
    else:
        sele = waters + " or organic"
    cmd.distance(
        "polar_contacts",
        sele,
        "sol",
        mode=2,
    )
    cmd.hide("labels")


def _fix_pymol_camera(active_site_center=None):
    from pymol import cmd

    # reset camera
    cmd.reset()
    if active_site_center is not None:
        cmd.center(active_site_center)
    else:
        cmd.center("waters")


def _determine_active_site_ids(active_site_ids: list[int]):
    from pymol import cmd

    selection = ""
    for i in active_site_ids:
        selection += str(i) + "+"
    selection = selection[: len(selection) - 1]
    selection = "resi " + selection
    aminokis_u_am: str = cmd.get_unused_name("active_site_aa")
    cmd.select(
        aminokis_u_am,
        " " + selection + " and polymer ",
    )
    cmd.show("licorice", aminokis_u_am)
    cmd.color("magenta", aminokis_u_am)
    # pseudoatom in active site center
    active_site_center = cmd.get_unused_name("activesite_center_")
    cmd.pseudoatom(active_site_center, aminokis_u_am)
    cmd.hide(representation="everything", selection=active_site_center)
    return active_site_center, aminokis_u_am


def _add_density_map(density_map: str):
    from pymol import cmd

    cmd.load(density_map)
    cmd.volume("water_density", density_map.split(".")[0])


def _add_crystal_waters(
    crystal_waters, protein, ligand_resname, dist, active_site_ids, active_site_center
):
    from pymol import cmd

    cmd.fetch(crystal_waters)
    cmd.hide("everything", crystal_waters)
    cmd.align(
        "polymer and " + crystal_waters,
        "polymer and " + protein,
    )
    if ligand_resname is not None:
        cmd.select(
            "crystal_waters",
            "("
            + crystal_waters
            + " and SOL) within "
            + str(dist)
            + " of resname "
            + ligand_resname,
        )
    elif active_site_ids is not None:
        cmd.select(
            "crystal_waters",
            "("
            + crystal_waters
            + " and SOL) within 10 of resname "
            + active_site_center,
        )
    else:
        cmd.select("crystal_waters", crystal_waters + " and SOL")
    cmd.show("spheres", "crystal_waters")
    cmd.set("sphere_scale", "0.4", "crystal_waters")
    if os.path.exists(crystal_waters + ".cif"):
        os.remove(crystal_waters + ".cif")


def _add_hydrogen_and_bond(wname, Hpos, Hname, resn, resi):
    from pymol import cmd

    cmd.pseudoatom(
        wname,
        pos=[Hpos[0], Hpos[1], Hpos[2]],
        name=Hname,
        resn=resn,
        resi=resi,
        elem="H",
        chain="W",
    )
    cmd.bond(f"{wname} and name O", f"{wname} and name {Hname}")


def _make_water_objects(water_type, waterO, waterH1, waterH2, output_file):
    from pymol import cmd

    cntr = {"FCW": 0, "WCW": 0, "HCW": 0}
    ind = 0  # initialize index
    while ind < len(water_type):
        tip, Opos, H1pos, H2pos = (
            water_type[ind],
            waterO[ind],
            waterH1[ind],
            waterH2[ind],
        )
        cntr[tip] += 1
        wname = tip + str(cntr[tip])
        resis = cmd.identify("all", mode=0)
        if len(resis) == 0:
            highest_resi = 0
        else:
            highest_resi = np.max(resis)
        if output_file is None or output_file.endswith(".pse"):
            resn = "SOL"
        elif output_file.endswith(".pdb"):
            resn = f"{tip}"
        cmd.create(wname, "none", source_state=0, target_state=0)
        cmd.pseudoatom(
            wname,
            pos=[Opos[0], Opos[1], Opos[2]],
            name="O",
            resn=resn,
            resi=highest_resi + 1,
            elem="O",
            chain="W",
        )
        if tip == "onlyO":
            cmd.show("spheres", wname)
            cmd.set("sphere_scale", 0.1, wname)
        else:
            _add_hydrogen_and_bond(wname, H1pos, "H1", resn, highest_resi + 1)
            _add_hydrogen_and_bond(wname, H2pos, "H2", resn, highest_resi + 1)
            # check future water type
            add_ind = 0
            ghind = 0
            while ind + add_ind + 1 < len(water_type) and tip != "FCW":
                if (
                    tip == "HCW"
                    and water_type[ind + add_ind + 1] == "HCW"
                    and np.array_equal(Opos, waterO[ind + add_ind + 1])
                ):
                    _add_hydrogen_and_bond(
                        wname,
                        waterH2[ind + add_ind + 1],
                        f"H{add_ind+3}",
                        resn,
                        highest_resi + 1,
                    )
                    add_ind += 1
                elif (
                    tip == "WCW"
                    and water_type[ind + add_ind + 1] == "WCW"
                    and np.array_equal(Opos, waterO[ind + add_ind + 1])
                ):
                    hind = 0
                    # check for all previos Hs
                    H1eq = [
                        True
                        if np.array_equal(waterH1[ind], waterH1[ind + dind + 1])
                        else False
                        for dind in range(add_ind)
                    ] + [
                        True
                        if np.array_equal(waterH2[ind], waterH1[ind + dind + 1])
                        else False
                        for dind in range(add_ind)
                    ]
                    if H1eq.count(True) == 0:
                        _add_hydrogen_and_bond(
                            wname,
                            waterH1[ind + add_ind + 1],
                            f"H{add_ind+ghind+3}",
                            resn,
                            highest_resi + 1,
                        )
                        hind += 1
                    H2eq = [
                        True
                        if np.array_equal(waterH1[ind], waterH2[ind + dind + 1])
                        else False
                        for dind in range(add_ind)
                    ] + [
                        True
                        if np.array_equal(waterH2[ind], waterH2[ind + dind + 1])
                        else False
                        for dind in range(add_ind)
                    ]
                    if H2eq.count(True) == 0:
                        _add_hydrogen_and_bond(
                            wname,
                            waterH2[ind + add_ind + 1],
                            f"H{add_ind+hind+ghind+3}",
                            resn,
                            highest_resi + 1,
                        )
                        hind += 1
                    if hind > 0:
                        add_ind += 1
                        ghind += hind
                else:
                    break
            ind += 1 + add_ind
            if output_file is None or output_file.endswith(".pse"):
                cmd.show("lines", wname)
                if tip == "FCW":
                    cmd.color("firebrick", wname)
                elif tip == "HCW":
                    cmd.color("skyblue", wname)
                    if add_ind > 0:
                        cmd.color("red", "name H1 and " + wname)
                elif tip == "WCW":
                    cmd.color("limegreen", wname)
            cmd.show("sticks", wname)


def visualise_pymol(
    water_type: list[str],
    waterO: list[list[float]],
    waterH1: list[list[float]],
    waterH2: list[list[float]],
    aligned_protein: str | None = "aligned.pdb",
    output_file: str | None = None,
    active_site_ids: list[int] | None = None,
    crystal_waters: str | None = None,
    ligand_resname: str | None = None,
    dist: float = 10.0,
    density_map: str | None = None,
    polar_contacts: bool = False,
    lunch_pymol: bool = True,
    reinitialize: bool = True,
) -> None:
    """Visualises results via `pymol <https://pymol.org/>`__.

    Visualises results using pymol in a pymol session or saves to a file.
    On mac OS the interactive pymol session will fail to lunch. If `output_file`
    is `None`, a visalisation state will be saved to
    `pymol_water_visualization.pse` in this case on mac OS.

    Args:
        water_type (list): List containing water type results from
            water clustering.
        waterO (list): Coordiantes of Oxygen atom in water molecules.
        waterH1 (list): Coordinates of Hydrogen1 atom in water molecules.
        waterH2 (list): Coordinates of Hydrogen2 atom in water molecules.
        aligned_protein (str | None, optional): file name containing protein
            configuration trajectory was aligned to. If `None` no
            protein will be shown. Defaults to "aligned.pdb".
        output_file (str | None, optional): File to save the
            visualisation state. If ``None``, a pymol session is started
            (this probably doesn't work on Mac OSX). Defaults to None.
        active_site_ids (list[int] | None, optional): Residue ids -
            numbers of aminoacids in active site. These are visualised
            as licorice. Defaults to None.
        crystal_waters (str | None, optional): PDBid from which crystal
            waters will attempted to be extracted. Defaults to None.
        ligand_resname (str | None, optional): Residue name of the
            ligand around which crystal waters (oxygens) shall be
            selected. Defaults to None.
        dist (float): distance from the centre of ligand around which
            crystal waters shall be selected. Defaults to 10.0.
        density_map (str | None, optional): Water density map to add to
            visualisation session (usually .dx file). Defaults to None.
        polar_contacts (bool, optional): If `True` polar contacts
            between waters and protein will be visualised. Defaults to
            False.
        lunch_pymol (bool, optional): If `True` pymol will be lunched
            in interactive mode. If `False` pymol will be imported
            without lunching. Defaults to True.
        reinitialize (bool, optional): If `True` pymol will be
            reinitialized (defaults restored and objects cleaned).
            Defaults to True.

    Example::

        # Read the results from files
        water_types, coord_O, coord_H1, coord_H2 = read_results(
            fname = "Clust_res.dat",
            typefname = "Type_Clust_res.dat",
        )
        visualis_pymol(
            water_type = water_types,
            waterO = coord_O,
            waterH1 = coord_H1,
            waterH2 = coord_H2,
            aligned_protein = "aligned.pdb",
            output_file = "results.pse",
            active_site_ids = [1,5,77,98],
            crystal_waters = "3t73",
            ligand_resname = "UBX",
            dist = 8.0,
        )
    """
    if platform.system() == "Darwin":
        lunch_pymol = False
    _initialize_pymol(reinitialize, lunch_pymol)
    if platform.system() == "Darwin" and output_file is None:
        import warnings

        warnings.warn(
            "mac OS detected interactive pymol session cannot be lunched. Visualisation state will be saved to pymol_water_visualization.pse",
            RuntimeWarning,
        )
        output_file = "pymol_water_visualization.pse"
    from pymol import cmd

    cmd.hide("everything")
    active_site_center = None
    aminokis_u_am = None
    if aligned_protein is not None:
        cmd.load(aligned_protein)
        cmd.hide("everything")
        # polymer for surface def
        tmpObj = cmd.get_unused_name("_tmp")
        cmd.create(tmpObj, "( all ) and polymer", zoom=0)
        # aminoacids in active site
        if active_site_ids is not None:
            aminokis_u_am, active_site_center = _determine_active_site_ids(
                active_site_ids
            )
        # protein surface
        _make_protein_surface_with_ligand()
    _make_water_objects(water_type, waterO, waterH1, waterH2, output_file)
    waters: str = cmd.get_unused_name("waters")
    cmd.select(waters, "SOL in (FCW* or WCW* or HCW*)")
    if polar_contacts:
        _add_polar_contacts(waters, aminokis_u_am)
    # Add crystal waters
    if crystal_waters and aligned_protein is not None:
        _add_crystal_waters(
            crystal_waters,
            aligned_protein,
            ligand_resname,
            dist,
            active_site_ids,
            active_site_center,
        )
    # add volume density visualisation
    if density_map is not None:
        _add_density_map(density_map)
    _fix_pymol_camera(active_site_center)
    # save
    if output_file is not None:
        cmd.save(output_file)


def visualise_pymol_from_pdb(
    pdbfile: str,
    active_site_ids: list[int] | None = None,
    crystal_waters: str | None = None,
    ligand_resname: str | None = None,
    dist: float = 10.0,
    density_map: str | None = None,
    polar_contacts: bool = False,
    lunch_pymol: bool = True,
    reinitialize: bool = True,
) -> None:
    """Make a `pymol <https://pymol.org/>`__ session from a pdb file.

    Visualises a pdb made by :py:meth:`make_results_pdb_MDA` file with water
    clustering results in pymol.

    Args:
        pdbfile (str): Name of the pdb file to read, should end in .pdb
        active_site_ids (list[int] | None, optional): Residue ids -
            numbers of aminoacids in active site. These are visualised
            as licorice. Defaults to None.
        crystal_waters (str | None, optional): PDBid from which crystal
            waters will attempted to be extracted. Defaults to None.
        ligand_resname (str | None, optional): Residue name of the
            ligand around which crystal waters (oxygens) shall be
            selected. Defaults to None.
        dist (float): distance from the centre of ligand around which
            crystal waters shall be selected. Defaults to 10.0.
        density_map (str | None, optional): Water density map to add to
            visualisation session (usually .dx file). Defaults to None.
        polar_contacts (bool, optional): If `True` polar contacts
            between waters and protein will be visualised. Defaults to
            False.
        lunch_pymol (bool, optional): If `True` pymol will be lunched
            in interactive mode. If `False` pymol will be imported
            without lunching. Defaults to True.
        reinitialize (bool, optional): If `True` pymol will be
            reinitialized (defaults restored and objects cleaned).
            Defaults to True.

    Example::

        visualise_pymol_from_pdb(
            pdbfile = "results.pdb",
            active_site_ids = [1,5,77,98],
            crystal_waters = "3t73",
            ligand_resname = "UBX",
            dist = 8.0,
            density_map = "waters.dx"
        )
    """
    if platform.system() == "Darwin":
        lunch_pymol = False
    _initialize_pymol(reinitialize, lunch_pymol)
    from pymol import cmd

    cmd.load(pdbfile)
    cmd.hide("everything")
    # polymer for surface def
    tmpObj = cmd.get_unused_name("_tmp")
    cmd.create(tmpObj, "( all ) and polymer", zoom=0)
    # aminoacids in active site
    active_site_center = None
    aminokis_u_am = None
    if active_site_ids is not None:
        aminokis_u_am, active_site_center = _determine_active_site_ids(active_site_ids)
    else:
        active_site_center = None
        aminokis_u_am = None
    # protein surface
    _make_protein_surface_with_ligand()
    # add water representations
    # conserved waters
    conserved = cmd.get_unused_name("FCW_")
    cmd.select(conserved, "resname FCW")
    cmd.color("firebrick", conserved)
    # half conserved waters
    half_conserved = cmd.get_unused_name("HCW_")
    cmd.select(half_conserved, "resname HCW")
    cmd.color("skyblue", half_conserved)
    # semi conserved waters
    semi_conserved = cmd.get_unused_name("WCW_")
    cmd.select(semi_conserved, "resname WCW")
    cmd.color("limegreen", semi_conserved)
    # all waters
    waters = cmd.get_unused_name("waters_")
    cmd.select(
        waters,
        semi_conserved + " or " + half_conserved + " or " + conserved,
    )
    cmd.show("sticks", waters)
    # Add crystal waters
    if crystal_waters is not None:
        _add_crystal_waters(
            crystal_waters,
            pdbfile.split(".")[0],
            ligand_resname,
            dist,
            active_site_ids,
            active_site_center,
        )
    if polar_contacts:
        _add_polar_contacts(waters, aminokis_u_am)
    # add volume density visualisation
    if density_map is not None:
        _add_density_map(density_map)
    _fix_pymol_camera(active_site_center)


def _initialize_pymol(reinitialize: bool, finish: bool):
    """Initializes pymol.

    Initializes pymol for visualisation. If `finish` is `True` pymol
    will be lunched in interactive mode. If `False` pymol will be
    imported without lunching.

    Args:
        reinitialize (bool): If `True` pymol will be
            reinitialized (defaults restored and objects cleaned).
            Defaults to False.
        finish (bool): If `True` pymol will be lunched
            in interactive mode. If `False` pymol will be
            imported without lunching. Defaults to True.
    """
    try:
        import pymol
        from pymol import cmd
    except ModuleNotFoundError:
        raise Exception("pymol not installed. Either install pymol or use nglview")
    if finish:
        pymol.finish_launching(["pymol", "-q"])
    if reinitialize:
        cmd.reinitialize()


def visualise_nglview(
    water_type: list[str],
    waterO: list[list[float]],
    waterH1: list[list[float]],
    waterH2: list[list[float]],
    aligned_protein: str = "aligned.pdb",
    active_site_ids: list[int] | None = None,
    crystal_waters: str | None = None,
    density_map_file: str | None = None,
) -> NGLWidget:
    """Creates `nglview <https://github.com/nglviewer/nglview>`__  visualisation widget for results.

    Starts a nglview visualisation instance from clustering results.

    Args:
        water_type (list): List containing water type results from
            water clustering.
        waterO (list): Coordiantes of Oxygen atom in water molecules.
        waterH1 (list): Coordinates of Hydrogen1 atom in water molecules.
        waterH2 (list): Coordinates of Hydrogen2 atom in water molecules.
        aligned_protein (str, optional): file name containing protein
            configuration trajectory was aligned to. Defaults to "aligned.pdb".
        active_site_ids (list[int] | None, optional): Residue ids -
            numbers of aminoacids in active site. These are visualised
            as licorice. Defaults to None.
        crystal_waters (str | None, optional): PDBid from which crystal
            waters will attempted to be extracted. Defaults to None.
        density_map (str | None, optional): Water density map to add to
            visualisation session (usually .dx file). Defaults to None.

    Returns:
        NGLWidget: Returns nglview Widget for visualisation of the
        results.

    Example::

        # read results and visualise them using nglview
        water_types, coord_O, coord_H1, coord_H2 = read_results(
            fname = "Clust_res.dat",
            typefname = "Type_Clust_res.dat",
        )
        view = visualise_nglview(
            water_type = water_types,
            waterO = coord_O,
            waterH1 = coord_H1,
            waterH2 = coord_H2,
            aligned_protein = "aligned.pdb",
            active_site_ids = [1,5,77,98],
            crystal_waters = "3t73",
        )
        # initialise widget
        view
    """
    try:
        import nglview as ngl
    except ModuleNotFoundError:
        raise Exception("nglview not installed. Either install pymol or nglview")

    if aligned_protein is not None:
        view: NGLWidget = ngl.show_file(aligned_protein, default_representation=False)
        view.clear_representations()
        view.add_representation("surface", selection="protein", opacity=0.5)
        view.add_representation("ball+stick", selection="water", color="red")
        selection = ""
        if active_site_ids is not None:
            for i in active_site_ids:
                selection = selection[: len(selection) - 3]
                selection += str(i) + " or "
            view.add_representation(
                "ball+stick",
                selection=selection,
                color="pink",
            )
    else:
        view = ngl.NGLWidget()
    col = {"FCW": "red", "WCW": "blue", "HCW": "green"}
    for tip, Opos, H1pos, H2pos in zip(water_type, waterO, waterH1, waterH2):
        view.add_pdbid("hoh")
        view[-1].add_representation("licorice", color=col[tip])
        view[-1].set_coordinates(np.asarray([H1pos, Opos, H2pos]))
    if crystal_waters is not None:
        view.add_pdbid(crystal_waters)
        view[-1].add_representation("spacefill", selection="water", color="red")
    if density_map_file is not None:
        view.add_component(density_map_file)
    return view
