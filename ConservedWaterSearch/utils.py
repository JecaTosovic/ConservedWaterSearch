from __future__ import annotations
import os
import platform
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from nglview import NGLWidget
    except ImportError:
        NGLWidget = None


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
    try:
        import pymol
        from pymol import cmd
    except ModuleNotFoundError:
        raise Exception("pymol not installed. Either install pymol or use nglview")
    if output_file is None and platform.system() != "Darwin" and lunch_pymol:
        pymol.finish_launching(["pymol", "-q"])
    if reinitialize:
        cmd.reinitialize()
    if platform.system() == "Darwin":
        import warnings

        warnings.warn(
            "mac OS detected interactive pymol session cannot be lunched. Visualisation state will be saved to pymol_water_visualization.pse",
            RuntimeWarning,
        )
        if output_file is None:
            output_file = "pymol_water_visualization.pse"
    cmd.hide("everything")
    if aligned_protein is not None:
        cmd.load(aligned_protein)
        cmd.hide("everything")
        # polymer for surface def
        tmpObj = cmd.get_unused_name("_tmp")
        cmd.create(tmpObj, "( all ) and polymer", zoom=0)
        # aminoacids in active site
        if active_site_ids:
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
        # cmd.show("sphere",active_site_center)
        # cmd.set ("sphere_scale",0.1,active_site_center)
        # protein surface
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
    cntr = {"FCW": 0, "WCW": 0, "HCW": 0}
    for tip, Opos, H1pos, H2pos in zip(water_type, waterO, waterH1, waterH2):
        cntr[tip] += 1
        wname = tip + str(cntr[tip])
        print(wname)
        cmd.fetch("hoh", wname)
        cmd.alter_state(
            0,
            wname,
            "(x,y,z)=(" + str(Opos[0]) + "," + str(Opos[1]) + "," + str(Opos[2]) + ")",
        )
        if tip == "onlyO":
            cmd.delete(wname + "and elem H")
        else:
            indiciesH: list[int] = []
            cmd.iterate_state(
                -1,
                wname + " and elem H",
                "indiciesH.append(index)",
                space={"indiciesH": indiciesH},
            )
            cmd.alter_state(
                0,
                wname + " and index " + str(indiciesH[0]),
                "(x,y,z)=("
                + str(H1pos[0])
                + ","
                + str(H1pos[1])
                + ","
                + str(H1pos[2])
                + ")",
            )
            cmd.alter_state(
                0,
                wname + " and index " + str(indiciesH[1]),
                "(x,y,z)=("
                + str(H2pos[0])
                + ","
                + str(H2pos[1])
                + ","
                + str(H2pos[2])
                + ")",
            )
            if output_file is None or output_file.endswith(".pse"):
                cmd.alter_state(
                    0,
                    wname,
                    "resn='SOL'",
                )
                cmd.show("lines", wname)
                if tip == "FCW":
                    cmd.color("firebrick", wname)
                elif tip == "HCW":
                    cmd.color("skyblue", wname)
                elif tip == "WCW":
                    cmd.color("limegreen", wname)
            elif output_file.endswith(".pdb"):
                cmd.alter_state(
                    0,
                    wname,
                    "resn='" + str(tip) + "'",
                )

        if tip == "onlyO":
            cmd.show("spheres", wname)
            cmd.set("sphere_scale", 0.1, wname)
        else:
            cmd.show("sticks", wname)
    waters: str = cmd.get_unused_name("waters")
    cmd.select(waters, "SOL in (FCW* or WCW* or HCW*)")
    if active_site_ids:
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
    # Add crystal waters
    if crystal_waters and aligned_protein is not None:
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
    # add volume density visualisation
    if density_map:
        cmd.load(density_map)
        cmd.volume("water_density", density_map.split(".")[0])
    # reset camera
    cmd.reset()
    if active_site_ids is not None:
        cmd.center(active_site_center)
    if os.path.exists("hoh.cif"):
        os.remove("hoh.cif")
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
    try:
        import pymol
        from pymol import cmd
    except ModuleNotFoundError:
        raise Exception("pymol not installed. Either install pymol or use nglview")
    if platform.system() != "Darwin":
        pymol.finish_launching(["pymol", "-q"])
    cmd.load(pdbfile)
    cmd.hide("everything")
    # polymer for surface def
    tmpObj = cmd.get_unused_name("_tmp")
    cmd.create(tmpObj, "( all ) and polymer", zoom=0)
    # aminoacids in active site
    if active_site_ids:
        aminokis_u_am = cmd.get_unused_name("active_site_aa")
        selection = ""
        for i in active_site_ids:
            selection += str(i) + "+"
        selection = selection[: len(selection) - 1]
        selection = "resi " + selection
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
    # cmd.show("sphere",active_site_center)
    # cmd.set ("sphere_scale",0.1,active_site_center)
    # protein surface
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
    # add water representations
    # conserved waters
    conserved = cmd.get_unused_name("FCW_")
    cmd.select(conserved, "resname FCW")
    cmd.show("lines", conserved)
    cmd.color("firebrick", conserved)
    # half conserved waters
    half_conserved = cmd.get_unused_name("HCW_")
    cmd.select(half_conserved, "resname HCW")
    cmd.show("lines", half_conserved)
    cmd.color("skyblue", half_conserved)
    # semi conserved waters
    semi_conserved = cmd.get_unused_name("WCW_")
    cmd.select(semi_conserved, "resname WCW")
    cmd.show("lines", semi_conserved)
    cmd.color("limegreen", semi_conserved)
    # all waters
    waters = cmd.get_unused_name("waters_")
    cmd.select(
        waters,
        semi_conserved + " or " + half_conserved + " or " + conserved,
    )
    # Add crystal waters
    if crystal_waters:
        cmd.fetch(crystal_waters)
        cmd.hide("everything", crystal_waters)
        cmd.align(
            "polymer and " + crystal_waters,
            "polymer and " + pdbfile.split(".")[0],
        )
        if ligand_resname:
            cmd.select(
                "crystal_waters",
                "("
                + crystal_waters
                + " and SOL) within "
                + str(dist)
                + " of resname "
                + ligand_resname,
            )
        else:
            cmd.select("crystal_waters", crystal_waters + " and SOL")
        cmd.show("spheres", "crystal_waters")
        cmd.set("sphere_scale", "0.4", "crystal_waters")
    # add volume density visualisation
    if density_map:
        cmd.load(density_map)
        cmd.volume("water_density", density_map.split(".")[0])
    # reset camera
    cmd.reset()
    if active_site_center is not None:
        cmd.center(active_site_center)
    # save
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
