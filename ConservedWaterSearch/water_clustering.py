from __future__ import annotations
from typing import TYPE_CHECKING


try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError:
    Axes = Figure = None

try:
    from nglview import NGLWidget
except ImportError:
    NGLWidget = None

if TYPE_CHECKING:
    pass

import os
import warnings
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_xi, HDBSCAN

from ConservedWaterSearch.hydrogen_orientation import (
    hydrogen_orientation_analysis,
)
from ConservedWaterSearch.utils import (
    read_results,
    visualise_nglview,
    visualise_pymol,
    _check_mpl_installation,
    _append_new_result,
)


class WaterClustering:
    """Class for performing water clustering.

    First, oxygens are clustered using OPTICS or HDBSCAN, followed by
    analysis of orientations for classification of waters into one of 3
    proposed conserved water types (for more information see
    :ref:`conservedwaters:theory, background and methods`):

        - FCW (Fully Conserved Water): hydrogens are strongly oriented in
          two directions with angle of 104.5
        - HCW (Half Conserved Water): one set (cluster) of hydrogens is
          oriented in a single direction and other hydrogen's
          orientations are spread into different orientations with angle
          of 104.5
        - WCW (Weakly Conserved Water): several orientation combinations
          exsist with satisfying water angles

    To run the calculation use either :py:meth:`multi_stage_reclustering`
    function to start Multi Stage ReClustering (MSRC) procedure or
    :py:meth:`single_clustering` to start a single clustering (SC) procedure.
    MSRC produces better results at the cost of computational time,
    while SC is very quick but results are worse and significant amount of
    waters might not be identified at all. For more details see
    :cite:`conservedwatersearch2022`.

    """

    def __init__(
        self,
        nsnaps: int,
        clustering_algorithm: str = "OPTICS",
        water_types_to_find: list[str] = ["FCW", "HCW", "WCW"],
        restart_after_found: bool = False,
        min_samples: list[int] = None,
        xis: list[float] = [
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00001,
        ],
        numbpct_oxygen: float = 0.8,
        normalize_orientations: bool = True,
        numbpct_hyd_orient_analysis: float = 0.85,
        kmeans_ang_cutoff: float = 120,
        kmeans_inertia_cutoff: float = 0.4,
        FCW_angdiff_cutoff: float = 5,
        FCW_angstd_cutoff: float = 17,
        other_waters_hyd_minsamp_pct: float = 0.15,
        nonFCW_angdiff_cutoff: float = 15,
        HCW_angstd_cutoff: float = 17,
        WCW_angstd_cutoff: float = 20,
        weakly_explained: float = 0.7,
        xiFCW: list[float] = [0.03],
        xiHCW: list[float] = [0.05, 0.01],
        xiWCW: list[float] = [0.05, 0.001],
        njobs: int = 1,
        verbose: int = 0,
        debugO: int = 0,
        debugH: int = 0,
        plotend: bool = False,
        plotreach: bool = False,
        restart_data_file: str | None = None,
        output_file: str | None = None,
    ) -> None:
        """Initialise :py:class:`WaterClustering` class.

        The input parameters determine the options for oxygen clustering and
        hydrogen orienataion analysis if applicable.

        Args:
            nsnaps (int): Number of trajectory snapshots related to
                the data set.
            clustering_algorithm (str, optional): Options are "OPTICS"
                or "HDBSCAN". OPTICS provides slightly better results,
                but is also slightly slower. Defaults to "OPTICS".
            water_types_to_find (list[str], optional): Defines which
                water types to search for. Any combination of "FCW",
                "HWC" and "WCW" is allowed, or "onlyO" for oxygen
                clustering only. Defaults to ["FCW", "HCW", "WCW"].
            restart_after_found (bool, optional): If ``True`` restarts
                clustering after each water is found. ``False`` will
                give the quick version of multi-stage reculstering
                approach. Defaults to False.
            min_samples (list[int], optional): List of minimum samples
                for OPTICS or HDBSCAN. If ``None`` following range is
                used ``[int(0.25 * nsnaps), nsnaps]`` is used. For single
                clustering users should provide a single integer between
                0 and ``nsnaps`` in a list. Defaults to None.
            xis (list[float], optional): List of xis for OPTICS
                clustering. This is ignored for HDBSCAN. Defaults to
                [ 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001,
                0.00001]. For single clustering users should provide a
                single float between 0 and 1 in a list.
            numbpct_oxygen (float, optional): Percentage of
                ``nsnaps`` required for oxygen cluster to be considered
                valid and water conserved. The check is enforced on
                the lower limit ``nsnaps * numbpct_oxygen`` as well as
                the upper limit ``nsnaps * (2-numbpct_oxygen)``.
                Defaults to 0.8.
            normalize_orientations (bool, optional): If orientations
                should be normalised to unit length or not. Defaults to True.
            numbpct_hyd_orient_analysis (float, optional): Minimum
                allowed size of the hydrogen orientation cluster.
                Defaults to 0.85.
            kmeans_ang_cutoff (float, optional): Maximum value of angle (in
                deg) allowed for FCW in kmeans clustering to be considered
                correct water angle. Defaults to 120.
            kmeans_inertia_cutoff (float, optional): upper limit allowed on
                kmeans inertia (measure of spread of data in a cluster).
                Defaults to 0.4.
            FCW_angdiff_cutoff (float, optional): Maximum value of angle (in
                deg) allowed for FCW in OPTICS/HDBSCAN clustering to be
                considered correct water angle. Defaults to 5.
            FCW_angstd_cutoff (float, optional): Maximal standard deviation
                of angle distribution of orientations of two hydrogens
                allowed for water to be considered FCW. Defaults to 17.
            other_waters_hyd_minsamp_pct (float, optional): Minimum samples to
                choose for OPTICS or HDBSCAN clustering as percentage of
                number of water molecules considered for HCW and WCW.
                Defaults to 0.15.
            nonFCW_angdiff_cutoff (float, optional): Maximum standard
                deviation of angle allowed for HCW and WCW to be considered
                correct water angle. Defaults to 15.
            HCW_angstd_cutoff (float, optional): Maximum standard deviation
                cutoff for WCW angles to be considered correct water angles.
                Defaults to 17.
            WCW_angstd_cutoff (float, optional): Maximum standard deviation
                cutoff for WCW angles to be considered correct water angles.
                Defaults to 20.
            weakly_explained (float, optional): percentage of explained
                hydrogen orientations for water to be considered WCW.
                Defaults to 0.7.
            xiFCW (list, optional): Xi value for OPTICS clustering for
                FCW. Don't touch this unless you know what you are
                doing. Defaults to [0.03].
            xiHCW (list, optional): Xi value for OPTICS clustering for
                HCW. Don't touch this unless you know what you are doing.
                Defaults to [0.05, 0.01].
            xiWCW (list, optional): Xi value for OPTICS clustering for
                WCW. Don't touch this unless you know what you are doing.
                Defaults to [0.05, 0.001].
            njobs (int, optional): how many cpu cores to use for clustering.
                Defaults to 1.
            verbose (int, optional): verbosity of output. Defaults to 0.
            debugH (int, optional): debug level for orientations. Defaults to 0.
            plotend (bool, optional): weather to plot everything at end
                of run. Defaults to False.
            plotreach (bool, optional): weather to plot the reachability
                plot for OPTICS when debuging. Defaults to False.
            restart_data_file (str, optional): Restart data file. If
                ``None`` restarting is not possible and no restart file
                is generated. Both ``restart_data_file`` and
                ``output_file`` have to be provided for clustering
                restarting. Defaults to None.
            output_file (str | None, optional): If ``None`` results are
                not saved to a file. If string is provided results
                (including temporary results) are saved to a file with
                that name. Both ``restart_data_file`` and
                ``output_file`` have to be provided for clustering
                restarting. Defaults to None.
        """
        if nsnaps <= 0:
            raise Exception(f"nsnaps must be positive {nsnaps}")
        if not isinstance(nsnaps, int):
            raise Exception(f"nsnaps must be an integer, but its {type(nsnaps)}")
        self.nsnaps: int = nsnaps
        self.clustering_algorithm = clustering_algorithm
        self.water_types_to_find = water_types_to_find
        self.restart_after_find = restart_after_found
        if min_samples is None:
            self.min_samples = self._check_and_setup_MSRC(0.25, 1)
        else:
            self.min_samples = min_samples
        self.xis = xis
        self.normalize_orientations: bool = normalize_orientations
        self.numbpct_oxygen = numbpct_oxygen
        self.numbpct_hyd_orient_analysis = numbpct_hyd_orient_analysis
        self.kmeans_ang_cutoff = kmeans_ang_cutoff
        self.kmeans_inertia_cutoff = kmeans_inertia_cutoff
        self.conserved_angdiff_cutoff = FCW_angdiff_cutoff
        self.conserved_angstd_cutoff = FCW_angstd_cutoff
        self.other_waters_hyd_minsamp_pct = other_waters_hyd_minsamp_pct
        self.noncon_angdiff_cutoff = nonFCW_angdiff_cutoff
        self.halfcon_angstd_cutoff = HCW_angstd_cutoff
        self.weakly_angstd_cutoff = WCW_angstd_cutoff
        self.weakly_explained = weakly_explained
        self.xiFCW = xiFCW
        self.xiHCW = xiHCW
        self.xiWCW = xiWCW
        self.njobs = njobs
        self.verbose = verbose
        self.debugO = debugO
        self.debugH = debugH
        self.plotreach = plotreach
        self.plotend = plotend
        self.restart_data_file = restart_data_file
        self.output_file: str | None = output_file
        if self.plotend:
            if not (self.debugH < 2 or self.debugO < 2):
                self.plotend = False
                warnings.warn(
                    "plotend set to True while debugH or debugO are >1; setting back to False"
                )
        self._waterO: list[np.ndarray] = []
        self._waterH1: list[np.ndarray] = []
        self._waterH2: list[np.ndarray] = []
        self._water_type: list[str] = []
        self._check_cls_alg_and_whichH()

    def run(self, oxygen_positions, hydrogen1_positions, hydrogen2_positions):
        """Run water clustering.

        Results will be stored in ``self.water_clusters``.

        Args:
            oxygen_positions (np.ndarray): Oxygen coordinates.
            hydrogen1_positions (np.ndarray): Hydrogen 1 orientations.
            hydrogen2_positions (np.ndarray): Hydrogen 2 orientations.
        """
        self._check_data(
            oxygen_positions,
            hydrogen1_positions,
            hydrogen2_positions,
        )
        self._scan_clustering_params(
            oxygen_positions, hydrogen1_positions, hydrogen2_positions
        )

    def multi_stage_reclustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray | None,
        H2: np.ndarray | None,
        clustering_algorithm: str = "OPTICS",
        lower_minsamp_pct: float = 0.25,
        every_minsamp: int = 1,
        xis: list[float] = [
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00001,
        ],
        whichH: list[str] = ["FCW", "HCW", "WCW"],
    ) -> None:
        """Multi Stage ReClustering (MSRC) procedure for obtaining conserved
        water molecules.

        Main loop - loops over water clustering parameter space
        (minsamp and xi) and clusters oxygens first - if a clustering
        with satisfactory oxygen clustering and hydrogen orientation
        clustering (optional) is found, elements of that water cluster
        are removed from the data set and water clustering starts from
        the beginning. Loops until no satisfactory clusterings are
        found. For more details see :cite:`conservedwatersearch2022`.

        Args:
            Odata (np.ndarray): Oxygen coordinates.
            H1 (np.ndarray | None): Hydrogen 1 orientations. If None ``whichH``
                must be "onlyO".
            H2 (np.ndarray | None): Hydrogen 2 orientations. If None ``whichH``
                must be "onlyO".
            clustering_algorithm (str, optional): Options are "OPTICS"
                or "HDBSCAN". OPTICS provides slightly better results, but
                is also slightly slower. Defaults to "OPTICS".
            lower_minsamp_pct (float, optional): Lowest minsamp value
                used for clustering. The range is from ``nsnaps``
                to ``lower_minsamp_pct`` times ``nsnaps``.
                Defaults to 0.25.
            every_minsamp (int, optional): Step for sampling of minsamp
                in range from ``nsnaps`` to ``lower_minsamp_pct`` times
                ``nsnaps``. If 1 uses all integer values in range.
                Defaults to 1.
            xis (list[float], optional): List of xis for OPTICS
                clustering. This is ignored for HDBSCAN. Defaults to
                [ 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001].
            whichH (list[str], optional): Defines which water types to
                search for. Any combination of "FCW", "HWC" and "WCW" is
                allowed, or "onlyO" for oxygen clustering only.
                Defaults to ["FCW", "HCW", "WCW"].
        """
        self.restart_after_find = True
        self.clustering_algorithm = clustering_algorithm
        self.xis = xis
        self.water_types_to_find = whichH
        self.min_samples = self._check_and_setup_MSRC(lower_minsamp_pct, every_minsamp)
        self._check_cls_alg_and_whichH()
        self.run(Odata, H1, H2)

    def quick_multi_stage_reclustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray | None,
        H2: np.ndarray | None,
        clustering_algorithm: str = "OPTICS",
        lower_minsamp_pct: float = 0.25,
        every_minsamp: int = 1,
        xis: list[float] = [
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00001,
        ],
        whichH: list[str] = ["FCW", "HCW", "WCW"],
    ) -> None:
        """Quick Multi Stage ReClustering (QMSRC) procedure for
        obtaining conserved water molecules.

        Main loop - loops over water clustering parameter space
        (minsamp and xi) and clusters oxygens first - clusters with
        satisfactory oxygen clustering and hydrogen orientation
        clustering (optional) are found, elements of those water cluster
        are added to the list of conserved waters. The data for those
        waters is removed from the data set but clustering does not
        restart.

        Args:
            Odata (np.ndarray): Oxygen coordinates.
            H1 (np.ndarray | None): Hydrogen 1 orientations. If None ``whichH``
                must be "onlyO".
            H2 (np.ndarray | None): Hydrogen 2 orientations. If None ``whichH``
                must be "onlyO".
            clustering_algorithm (str, optional): Options are "OPTICS"
                or "HDBSCAN". OPTICS provides slightly better results, but
                is also slightly slower. Defaults to "OPTICS".
            lower_minsamp_pct (float, optional): Lowest minsamp value
                used for clustering. The range is from ``nsnaps``
                to ``lower_minsamp_pct`` times ``nsnaps``.
                Defaults to 0.25.
            every_minsamp (int, optional): Step for sampling of minsamp
                in range from ``nsnaps`` to ``lower_minsamp_pct`` times
                ``nsnaps``. If 1 uses all integer values in range.
                Defaults to 1.
            xis (list[float], optional): List of xis for OPTICS
                clustering. This is ignored for HDBSCAN. Defaults to
                [ 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001,
                0.00001].
            whichH (list[str], optional): Defines which water types to
                search for. Any combination of "FCW", "HWC" and "WCW" is
                allowed, or "onlyO" for oxygen clustering only.
                Defaults to ["FCW", "HCW", "WCW"].
        """
        self.restart_after_find = False
        self.clustering_algorithm = clustering_algorithm
        self.xis = xis
        self.water_types_to_find = whichH
        self.min_samples = self._check_and_setup_MSRC(lower_minsamp_pct, every_minsamp)
        self._check_cls_alg_and_whichH()
        self.run(Odata, H1, H2)

    def single_clustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray | None,
        H2: np.ndarray | None,
        clustering_algorithm: str = "OPTICS",
        minsamp: int | None = None,
        xi: float | None = None,
        whichH: list[str] = ["FCW", "HCW", "WCW"],
    ) -> None:
        """Single clustering procedure.

        In single clustering procedure oxygen clustering is run only
        once with given ``minsamp`` and ``xi`` (if applicable - only for
        OPTICS).

        Args:
            Odata (np.ndarray): Oxygen coordinates.
            H1 (np.ndarray | None): Hydrogen 1 orientations. If None ``whichH``
                must be "onlyO".
            H2 (np.ndarray | None): Hydrogen 2 orientations. If None ``whichH``
                must be "onlyO".
            clustering_algorithm (str, optional): Options are "OPTICS"
                or "HDBSCAN". OPTICS provides slightly better results, but
                is also slightly slower. Defaults to "OPTICS".
            minsamp (int | None, optional): Minimum samples parameter
                for OPTICS or HDBSCAN. If None ``numbpct_oxygen`` *
                ``nsnaps`` is used. Defaults to None.
            xi (float | None, optional): Xi value for OPTICS. If
                ``None`` value of 0.05 is used. If
                ``clustering_algorithm`` is HDBSCAN its ignored.
                Defaults to None.
            whichH (list[str], optional): Defines which water types to
                search for. Any combination of "FCW", "HWC" and "WCW" is
                allowed, or "onlyO" for oxygen clustering only.
                Defaults to ["FCW", "HCW", "WCW"].
        """
        self.restart_after_find = False
        self.clustering_algorithm = clustering_algorithm
        self.water_types_to_find = whichH
        self._check_cls_alg_and_whichH()
        self.min_samples, self.xis = self._check_and_setup_single(xi, minsamp)
        self.run(Odata, H1, H2)

    def save_results(self, file_name: str) -> None:
        """Saves clustering results and paramters to a file.

        Top of the results file contains clustering parametrs after
        which results are saved in the same file.

        Args:
            file_name str: File name of the file that will contain results.
        """
        self._save_clustering_options(file_name)
        for i in range(len(self._waterO)):
            if len(self._waterH1) == 0 and len(self._waterH2) == 0:
                _append_new_result(
                    self._water_type[i], self._waterO[i], None, None, file_name
                )
            else:
                _append_new_result(
                    self._water_type[i],
                    self._waterO[i],
                    self._waterH1[i],
                    self._waterH2[i],
                    file_name,
                )

    def restart_cluster(
        self,
        partial_results_file: str,
        partial_data_file: str,
    ) -> None:
        """Read the clustering options and intermediate results from a
        file and restart the clustering procedure.

        Args:
            partial_data_file str: File name of the file containing
                intermediate set of data of hydrogen and oxygen
                coordinates.
            partial_results_file str: File name containing partial
                results with determined water coordinates.
        """
        if os.path.isfile(partial_data_file):
            data: np.ndarray = np.loadtxt(partial_data_file)
            if data.shape[1] == 3:
                Odata: np.ndarray = data
                H1: None | np.ndarray = None
                H2: None | np.ndarray = None
            else:
                Odata = data[:, :3]
                H1 = data[:, 3:6]
                H2 = data[:, 6:9]
        else:
            raise Exception("data file not found")
        if os.path.isfile(partial_results_file):
            self.read_and_set_water_clust_options(partial_results_file)
            self._water_type, self._waterO, self._waterH1, self._waterH2 = read_results(
                partial_results_file
            )
        else:
            raise Exception("partial results file not found")
        self.run(Odata, H1, H2)

    def read_and_set_water_clust_options(self, file_name: str) -> None:
        """Reads all class clustering options from save file and sets
        the parameters. Reads all parameters except clustering protocol
        and protocol parameters.

        Args:
            file_name str: Results or partial results file from which
                procedure parameters will be read.
        """
        if os.path.isfile(file_name):
            with open(file_name, "r") as f:
                lines: list[str] = f.read().splitlines()
                self.nsnaps = int(lines[0].strip())
                self.clustering_algorithm = lines[1].strip(" ")
                self.water_types_to_find = [i for i in lines[2].split(" ")]
                self.restart_after_find = lines[3] == "True"
                self.min_samples = [int(i) for i in lines[4].split(" ")]
                self.xis = [float(i) for i in lines[5].split(" ")]
                self.numbpct_oxygen = float(lines[6])
                self.normalize_orientations = lines[7] == "True"
                self.numbpct_hyd_orient_analysis = float(lines[8])
                self.kmeans_ang_cutoff = float(lines[9])
                self.kmeans_inertia_cutoff = float(lines[10])
                self.conserved_angdiff_cutoff = float(lines[11])
                self.conserved_angstd_cutoff = float(lines[12])
                self.other_waters_hyd_minsamp_pct = float(lines[13])
                self.noncon_angdiff_cutoff = float(lines[14])
                self.halfcon_angstd_cutoff = float(lines[15])
                self.weakly_angstd_cutoff = float(lines[16])
                self.weakly_explained = float(lines[17])
                self.xiFCW = [float(i) for i in lines[18].split(" ")]
                self.xiHCW = [float(i) for i in lines[19].split(" ")]
                self.xiWCW = [float(i) for i in lines[20].split(" ")]
                self.njobs = int(lines[21])
                self.verbose = int(lines[22])
                self.debugO = int(lines[23])
                self.debugH = int(lines[24])
                self.plotreach = lines[25] == "True"
                self.plotend = lines[26] == "True"
        else:
            raise Exception("output file not found")

    @classmethod
    def create_from_file(
        cls,
        file_name: str,
    ) -> WaterClustering:
        """Create a WaterClustering class from saved clustering options
        file or full or partial results file.

        Args:
            file_name str: Results or partial results file from which
                procedure parameters will be read.

        Returns:
            creates an instance of :py:class:`WaterClustering`
            class by reading options from a file.
        """
        instance = cls(1)
        instance.read_and_set_water_clust_options(file_name)
        return instance

    @classmethod
    def create_from_files_and_restart(
        cls, partial_output: str, partial_data_file: str
    ) -> WaterClustering:
        """Create a WaterClustering class from saved clustering restart
        and partial results files and restart clustering.

        Args:
            partial_file_name str: Partial results file from which
                procedure parameters will be read.
            partial_data_file str: Partial data file from which
                data will be read.

        Returns:
            creates an instance of :py:class:`WaterClustering`
            class and restarts clustering
        """
        instance = cls(1)
        instance.restart_cluster(partial_output, partial_data_file)
        return instance

    def visualise_pymol(
        self,
        aligned_protein: str | None = None,
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
        """Visualise results using `pymol <https://pymol.org/>`__.

        Args:
            aligned_protein (str, optional): file name containing protein
                configuration trajectory was aligned to. If ``None``
                only waters are shown. Defaults to None.
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
                visualisation session (usually .dx file). Defaults to
                None.
            polar_contacts (bool, optional): If `True` polar contacts
                between waters and protein will be visualised. Defaults to
                False.
            lunch_pymol (bool, optional): If `True` pymol will be lunched
                in interactive mode. If `False` pymol will be imported
                without lunching. Defaults to True.
            reinitialize (bool, optional): If `True` pymol will be
                reinitialized (defaults restored and objects cleaned).
                Defaults to True.
        """
        visualise_pymol(
            self._water_type,
            self._waterO,
            self._waterH1,
            self._waterH2,
            aligned_protein=aligned_protein,
            output_file=output_file,
            active_site_ids=active_site_ids,
            crystal_waters=crystal_waters,
            ligand_resname=ligand_resname,
            dist=dist,
            density_map=density_map,
            polar_contacts=polar_contacts,
            lunch_pymol=lunch_pymol,
            reinitialize=reinitialize,
        )

    def visualise_nglview(
        self,
        aligned_protein: str | None = None,
        active_site_ids: list[int] | None = None,
        crystal_waters: str | None = None,
        density_map: str | None = None,
    ) -> NGLWidget:
        """Visualise the results using `nglview <https://github.com/nglviewer/nglview>`__.

        Args:
            aligned_protein (str, optional): File containing protein
                configuration the original trajectory was aligned to.
                Defaults to ``None``.
            active_site_ids (list[int] | None, optional): Residue ids -
                numbers of aminoacids in active site. These are visualised
                as licorice. Defaults to None.
            crystal_waters (str | None, optional): PDBid from which crystal
                waters will attempted to be extracted. Defaults to None.
            density_map (str | None, optional): Water density map to add to
                visualisation session (usually .dx file). Defaults to None.

        Returns:
            NGLWidget: returns nglview instance widget which can be run
            in Ipyhon/Jupyter to create a visualisation instance
        """
        return visualise_nglview(
            self._water_type,
            self._waterO,
            self._waterH1,
            self._waterH2,
            aligned_protein=aligned_protein,
            active_site_ids=active_site_ids,
            crystal_waters=crystal_waters,
            density_map_file=density_map,
        )

    @property
    def water_type(self) -> list[str]:
        """List containing conserved water type classifications.

        Contains conserved water type classifications in the same order
        as coordinates in ``waterO`` and ``waterH1`` and ``waterH2``.
        Water types:

        - FCW (Fully Conserved Water): hydrogens are strongly oriented in
          two directions with angle of 104.5
        - HCW (Half Conserved Water): one set (cluster) of hydrogens is
          oriented in certain directions and other are spread into different
          orientations with angle of 104.5
        - WCW (Weakly Conserved Water): several orientation combinations
          exsist with satisfying water angles

        For more information see :cite:`conservedwatersearch2022` and
        :ref:`conservedwaters:theory, background and methods`).

        Returns:
            list[str]: Returns a list of strings containing water type
            classification - "FCW" or "HCW" or "WCW". If "onlyO", only
            oxygen clustering was performed.
        """
        return self._water_type

    @property
    def waterO(self) -> list[np.ndarray]:
        """Contains coordiantes of Oxygens of water molecules classified
        with water clustering.

        Returns:
            list[np.ndarray]: Returns a list of 3D xyz
            coordinates of oxygen positions in space
        """
        return self._waterO

    @property
    def waterH1(self) -> list[np.ndarray]:
        """Contains coordinates of first Hydrogen atom of water
        molecules classified with water clustering.

        Returns:
            list[np.ndarray]: Returns a list of 3D xyz
            coordinates of first hydrogens' positions in space
        """
        return self._waterH1

    @property
    def waterH2(self) -> list[np.ndarray]:
        """Contains coordinates of second Hydrogen atom of water
        molecules classified with water clustering.

        Returns:
            list[np.ndarray]: Returns a list of 3D xyz
            coordinates of second hydrogens' positions in space
        """
        return self._waterH2

    @property
    def water_clusters(self) -> list[dict]:
        """A single list containing main results.

        List of dicts containing coordinates of oxygen and two hydrogens
        and water classification. Each element in the list is a
        dictionary that contains keys "O", "H1", "H2" and "type" which
        correspond to oxygen coordinates, hydrogen 1 coordinates,
        hydrogen 2 coordinates and water classification respectively.
        """
        water_clusters = []
        for i in range(len(self._waterO)):
            if len(self._waterH1) > 0:
                water_clusters.append(
                    {
                        "O": self._waterO[i],
                        "H1": self._waterH1[i],
                        "H2": self._waterH2[i],
                        "type": self._water_type[i],
                    }
                )
            else:
                water_clusters.append(
                    {
                        "O": self._waterO[i],
                        "type": self._water_type[i],
                    }
                )
        return water_clusters

    def _scan_clustering_params(
        self,
        Odata,
        H1=None,
        H2=None,
    ):
        if self.output_file is not None:
            self._save_clustering_options(self.output_file)
        for wt in self.water_types_to_find:
            found: bool = False if len(Odata) < self.nsnaps else True
            while found:
                found = False
                # loop over minsamps- from N(snapshots) to 0.75*N(snapshots)
                for i in self.min_samples:
                    if self.clustering_algorithm == "OPTICS":
                        clust: OPTICS | HDBSCAN = OPTICS(
                            min_samples=int(i), n_jobs=self.njobs
                        )  # type: ignore
                        clust.fit(Odata)
                    # loop over xi
                    for j in self.xis:
                        # recalculate reachability - OPTICS reachability has to be recaculated when changing minsamp
                        if self.clustering_algorithm == "HDBSCAN":
                            clust = HDBSCAN(
                                min_cluster_size=int(self.nsnaps * self.numbpct_oxygen),
                                min_samples=int(i),
                                max_cluster_size=int(
                                    self.nsnaps * (2 - self.numbpct_oxygen)
                                ),
                                cluster_selection_method="eom",
                                n_jobs=self.njobs,
                                allow_single_cluster=True
                                if len(Odata) < self.nsnaps * 2
                                else False,
                            )
                            clust.fit(Odata)
                            clusters: np.ndarray = clust.labels_
                        elif self.clustering_algorithm == "OPTICS":
                            clusters = cluster_optics_xi(
                                reachability=clust.reachability_,  # type: ignore
                                predecessor=clust.predecessor_,  # type: ignore
                                ordering=clust.ordering_,  # type: ignore
                                min_samples=i,
                                xi=j,
                            )[0]
                        # Debug stuff
                        if self.debugO > 0:
                            dbgt: str = ""
                            if self.verbose > 0:
                                (aa, bb) = np.unique(clusters, return_counts=True)
                                dbgt = (
                                    f"Oxygen clustering {type(clust)} minsamp={i}, xi={j}, {len(np.unique(clusters[clusters!=-1]))} clusters \n"
                                    f"Required N(elem) range:{self.nsnaps*self.numbpct_oxygen:.2f} to {(2-self.numbpct_oxygen)*self.nsnaps}; (tar cls size={self.nsnaps} and numbpct={self.numbpct_oxygen:.2f})\n"
                                    f"N(elements) for each cluster: {bb}\n"
                                )
                                print(dbgt)
                            ff: Figure = _oxygen_clustering_plot(
                                Odata, clust, dbgt, self.debugO, self.plotreach
                            )
                        waters, idcs = self._analyze_oxygen_clustering(
                            Odata,
                            H1,
                            H2,
                            clusters,
                            [wt],
                        )
                        if self.debugO == 1:
                            plt = _check_mpl_installation()
                            plt.close(ff)
                        if len(waters) > 0:
                            found = True
                            if wt == "onlyO":
                                Odata, _, _ = self._delete_data(idcs, Odata)
                            else:
                                Odata, H1, H2 = self._delete_data(idcs, Odata, H1, H2)
                            self._add_water_solutions(waters)
                            if self.restart_data_file is not None:
                                self._save_intermediate_data(Odata, H1, H2)
                            i = i - 1
                            break
                    if (found and self.restart_after_find) or len(Odata) < self.nsnaps:
                        break
                # check if size of remaining data set is bigger then number of snapshots
                if len(Odata) < self.nsnaps or self.restart_after_find is False:
                    break
        if (self.debugH == 1 or self.debugO == 1) and self.plotend:
            plt = _check_mpl_installation()
            plt.show()

    def _analyze_oxygen_clustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray | None,
        H2: np.ndarray | None,
        clusters: np.ndarray,
        whichH: list[str],
    ) -> tuple[list[np.ndarray], list[int]]:
        """Helper function for analysing oxygen clustering and invoking
        hydrogen orientation clustering.

        Analyzes clusters for oxygen clustering. For oxygen clusters
        which have the size around number of samples, the hydrogen
        orientation analysis is performed and type of water molecule and
        coordinates are returned.

        Args:
            Odata (np.ndarray): Oxygen coordinates
            H1 (np.ndarray | None): Hydrogen 1 orientations. If None ``whichH``
                must be "onlyO".
            H2 (np.ndarray | None): Hydrogen 2 orientations. If None ``whichH``
                must be "onlyO".
            clusters (np.ndarray):  Output of clustering
                results from OPTICS or HDBSCAN.

        Returns:
            tuple[list[np.ndarray], list[int]]:
            returns two lists. First list contains valid conserved waters
            found. Each entry in the list is a list which contains the
            positions of oxygen, 2 hydrogen positions and water type
            found, the second list is a list of lists which contain
            arguments to be deleted if ``stop_after_frist_water_found``
            is True, else the second list is empty.
        """
        cluster_ids = np.unique(clusters[clusters != -1])
        cluster_ids.sort()
        min_neioc = self.nsnaps * self.numbpct_oxygen
        max_neioc = self.nsnaps * (2 - self.numbpct_oxygen)
        waters = []
        # make empty numpy array of integers
        idcs = np.array([], dtype=int)
        # Loop over all oxygen clusters (-1 is non cluster)
        for k in cluster_ids:
            mask = clusters == k
            # Number of elements in oxygen cluster
            neioc = np.count_nonzero(mask)
            # If number of elements in oxygen cluster is  Nsnap*0.85<Nelem<Nsnap*1.15 then ignore
            if min_neioc < neioc < max_neioc:
                if self.verbose > 0:
                    print(f"O clust {k}, size {len(clusters[clusters==k])}\n")
                O_center = np.mean(Odata[mask], axis=0)
                if "onlyO" not in self.water_types_to_find:
                    # Construct array of hydrogen orientations
                    orientations = np.vstack([H1[mask], H2[mask]])
                    # Analyse clustering with hydrogen orientation analysis and more debug stuff
                    hyd = hydrogen_orientation_analysis(
                        orientations,
                        self.numbpct_hyd_orient_analysis,
                        self.kmeans_ang_cutoff,
                        self.kmeans_inertia_cutoff,
                        self.conserved_angdiff_cutoff,
                        self.conserved_angstd_cutoff,
                        self.other_waters_hyd_minsamp_pct,
                        self.noncon_angdiff_cutoff,
                        self.halfcon_angstd_cutoff,
                        self.weakly_angstd_cutoff,
                        self.weakly_explained,
                        self.xiFCW,
                        self.xiHCW,
                        self.xiWCW,
                        self.njobs,
                        self.verbose,
                        self.debugH,
                        self.plotreach,
                        whichH,
                        self.normalize_orientations,
                    )
                    if self.plotreach and self.debugH > 0:
                        plt = _check_mpl_installation()
                        plt.show()
                    if len(hyd) > 0:
                        # add water atoms for pymol visualisation
                        for i in hyd:
                            water = [O_center]
                            water.append(O_center + i[0])
                            water.append(O_center + i[1])
                            water.append(i[2])
                            waters.append(water)
                        idcs = np.append(idcs, np.argwhere(mask).flatten())
                        # debug
                        if (
                            self.debugO == 1
                            and self.plotreach == 0
                            and self.debugH == 0
                        ):
                            plt = _check_mpl_installation()
                            plt.show()
                        if self.restart_after_find:
                            return waters, idcs
                else:
                    water = [O_center, "O_clust"]
                    waters.append(water)
                    idcs = np.append(idcs, np.argwhere(mask).flatten())
                    if self.restart_after_find:
                        return waters, idcs
        return waters, idcs

    def _save_clustering_options(self, fname: str) -> None:
        """Function that saves clustering options to a file.

        In order to restart the clustering procedure the intermediate
        results and clustering options need to be known. This function
        enables one to save the clustering options upon the start of
        clustering procedure. This happens automatically depending on
        ``save_intermediate_results`` parameter.

            fname (str, optional): file name to save clustering options to.
        """
        if fname is None:
            fname = self.output_file
        with open(fname, "w") as f:
            print(self.nsnaps, file=f)
            print(self.clustering_algorithm, file=f)
            print(*self.water_types_to_find, file=f)
            print(self.restart_after_find, file=f)
            print(*self.min_samples, file=f)
            print(*self.xis, file=f)
            print(self.numbpct_oxygen, file=f)
            print(self.normalize_orientations, file=f)
            print(self.numbpct_hyd_orient_analysis, file=f)
            print(self.kmeans_ang_cutoff, file=f)
            print(self.kmeans_inertia_cutoff, file=f)
            print(self.conserved_angdiff_cutoff, file=f)
            print(self.conserved_angstd_cutoff, file=f)
            print(self.other_waters_hyd_minsamp_pct, file=f)
            print(self.noncon_angdiff_cutoff, file=f)
            print(self.halfcon_angstd_cutoff, file=f)
            print(self.weakly_angstd_cutoff, file=f)
            print(self.weakly_explained, file=f)
            print(*self.xiFCW, file=f)
            print(*self.xiHCW, file=f)
            print(*self.xiWCW, file=f)
            print(self.njobs, file=f)
            print(self.verbose, file=f)
            print(self.debugO, file=f)
            print(self.debugH, file=f)
            print(self.plotreach, file=f)
            print(self.plotend, file=f)

    def _delete_data(
        self,
        elements: np.ndarray,
        Odata: np.ndarray,
        H1: None | np.ndarray = None,
        H2: None | np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """A helper function for deleting data from the dataset during
        MSRC procedure.

        Args:
            elements (np.ndarray): Indices to delete.
            Odata (np.ndarray): Oxygen data set array of
                Oxygen coordinates which will be cut down.
            H1 (None | np.ndarray, optional): Hydrogen 1
                data set array that contains orientations. Defaults to None.
            H2 (None | np.ndarray, optional): Hydrogen 2
                data set array that contains orientations. Defaults to None.


        Returns:
            tuple[ np.ndarray, np.ndarray | None, np.ndarray | None, ]:
            returns a new set of Oxygen and Hydrogen xyz coordinates array
            with some rows deleted.
        """
        Odata = np.delete(Odata, elements, 0)
        if H1 is not None:
            H1 = np.delete(H1, elements, 0)
        if H2 is not None:
            H2 = np.delete(H2, elements, 0)
        return Odata, H1, H2

    def _check_cls_alg_and_whichH(self):
        if (
            self.clustering_algorithm != "OPTICS"
            and self.clustering_algorithm != "HDBSCAN"
        ):
            raise Exception("clustering algorithm must be OPTICS or HDBSCAN")
        for i in self.water_types_to_find:
            if i not in ["FCW", "HCW", "WCW", "onlyO"]:
                raise Exception(
                    "whichH supports onlyO or any combination of FCW, HCW and WCW"
                )
        if "onlyO" in self.water_types_to_find and len(self.water_types_to_find) > 1:
            raise Exception("onlyO cannot be used with other water types")
        if self.clustering_algorithm == "OPTICS":
            for i in self.xis:
                if not isinstance(i, float):
                    raise Exception("xis must contain floats")
                if i > 1 or i < 0:
                    raise Exception("xis should be between 0 and 1")
        elif self.clustering_algorithm == "HDBSCAN":
            self.xis = [0.0]
        # sort min_samples in descending order
        if len(self.min_samples) > 1:
            self.min_samples = sorted(self.min_samples, reverse=True)

    def _check_and_setup_single(self, xis, minsamp):
        if minsamp is None:
            minsamp = int(self.numbpct_oxygen * self.nsnaps)
        elif not isinstance(minsamp, int):
            raise Exception("minsamp must be an int")
        elif minsamp > self.nsnaps or minsamp <= 0:
            raise Exception("minsamp must be between 0 and nsnaps")
        if xis is None:
            xis = 0.05
        elif not isinstance(xis, float):
            raise Exception("xi must be a float")
        elif xis < 0 or xis > 1:
            raise Exception("xis should be between 0 and 1")
        return [minsamp], [xis]

    def _check_and_setup_MSRC(self, lower_minsamp_pct, every_minsamp):
        if lower_minsamp_pct > 1.0000001 or lower_minsamp_pct < 0:
            raise Exception("lower_misamp_pct must be between 0 and 1")
        if not isinstance(every_minsamp, int):
            raise Exception("every_minsamp must be integer")
        if every_minsamp <= 0 or every_minsamp > self.nsnaps:
            raise Exception("every_minsamp must be  0<every_minsamp<=nsnaps")
        minsamps: list = list(
            reversed(
                range(
                    int(self.nsnaps * lower_minsamp_pct),
                    self.nsnaps + 1,
                    every_minsamp,
                )
            )
        )
        return minsamps

    def _check_data(self, Odata, H1, H2):
        if (H1 is None or H2 is None) and "onlyO" not in self.water_types_to_find:
            raise Exception(
                f"H1 and H2 have to be provided for non oxygen only search. Run type {self.water_types_to_find}"
            )
        if H1 is not None and H2 is not None:
            if len(Odata) != len(H1) or len(Odata) != len(H2) or len(H1) != len(H2):
                raise Exception("Odata, H1 and H2 have to be of same length")

    def _save_intermediate_data(self, Oxygen, H1, H2) -> None:
        if self.restart_data_file is not None:
            if H1 is not None and H2 is not None:
                np.savetxt(self.restart_data_file, np.c_[Oxygen, H1, H2])
            else:
                np.savetxt(self.restart_data_file, np.c_[Oxygen])

    def _add_water_solutions(
        self,
        waters: list,
    ) -> None:
        """A helper function which extends the solutions obtained from
        analysing hydrogen orientations.

        Args:
            waters (list): List containing results - coordinates of
                oxygens and two hydrogens and water classification.
        """
        for i in waters:
            O_coord = i[0]
            tip = i[-1]
            if len(i) > 2:
                H1 = i[1]
                H2 = i[2]
                self._waterH1.append(H1)
                self._waterH2.append(H2)
            else:
                H1 = None
                H2 = None
            if self.output_file is not None:
                _append_new_result(
                    tip,
                    O_coord,
                    H1,
                    H2,
                    self.output_file,
                )
            self._waterO.append(O_coord)
            self._water_type.append(tip)


def _oxygen_clustering_plot(
    Odata: np.ndarray,
    cc: OPTICS,
    title: str,
    debugO: int,
    plotreach: bool = False,
) -> Figure:
    """Function for plotting oxygen clustering results.

    For debuging oxygen clustering. Not ment for general usage.

    """
    if type(cc) is not OPTICS:
        plotreach = False
    if debugO > 0:
        plt = _check_mpl_installation()
        fig: Figure = plt.figure()
        if plotreach:
            ax: Axes = fig.add_subplot(1, 2, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        for k in np.sort(np.unique(cc.labels_)):
            jaba = Odata[cc.labels_ == k]
            s = 10
            if k == -1:
                s = 0.25
            ax.scatter(
                jaba[:, 0],
                jaba[:, 1],
                jaba[:, 2],
                label=f"{k} ({len(cc.labels_[cc.labels_==k])})",
                s=s,
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.legend()
        if plotreach:
            plt = _check_mpl_installation()
            lblls = cc.labels_[cc.ordering_]
            ax = fig.add_subplot(1, 2, 2)
            plt.gca().set_prop_cycle(None)
            space = np.arange(len(Odata))
            ax.plot(space, cc.reachability_)
            for clst in np.unique(lblls):
                if clst == -1:
                    ax.plot(
                        space[lblls == clst],
                        cc.reachability_[lblls == clst],
                        label=f"{clst} ({len(space[lblls==clst])})",
                        color="blue",
                    )
                else:
                    ax.plot(
                        space[lblls == clst],
                        cc.reachability_[lblls == clst],
                        label=f"{clst} ({len(space[lblls==clst])})",
                    )
            ax.legend()
    if debugO == 2:
        plt = _check_mpl_installation()
        plt.show()
    return fig
