from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from nglview import NGLWidget
    from io import TextIOWrapper
    from hdbscan import HDBSCAN

import os
import warnings

import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_xi

from ConservedWaterSearch.hydrogen_orientation import (
    hydrogen_orientation_analysis,
)
from ConservedWaterSearch.utils import (
    read_results,
    visualise_nglview,
    visualise_pymol,
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
          oriented in certain directions and other are spread into different
          orientations with angle of 104.5
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
        numbpct_oxygen: float = 0.8,
        normalize_orientations: bool = True,
        save_intermediate_results: bool = True,
        save_results_after_done: bool = True,
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
    ) -> None:
        """Initialise :py:class:`WaterClustering` class.

        The input parameters determine the options for oxygen clustering and
        hydrogen orienataion analysis if applicable.

        Args:
            nsnaps (int): Number of trajectory snapshots related to
                the data set.
            numbpct_oxygen (float, optional): Percentage of
                ``nsnaps`` required for oxygen cluster to be considered
                valid and water conserved. The check is enforced on
                the lower limit ``nsnaps * numbpct_oxygen`` as well as
                the upper limit ``nsnaps * (2-numbpct_oxygen)``.
                Defaults to 0.8.
            normalize_orientations (bool, optional): If orientations
                should be normalised to unit length or not. Defaults to True.
            save_intermediate_results (bool, optional): If ``True``
                intermediate results are saved throught the calculation
                procedure. Defaults to True.
            save_results_after_done (bool, optional): If ``True`` saves
                the results after the calculation has finished to files
                "Clustering_results.dat" and "Type_Clustering_results.dat".
                Defaults to True.
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
            plotreach (bool, optional): weather to plot the reachability
                plot for OPTICS when debuging. Defaults to False.
            which (list, optional): list of strings denoting which water
                types to search for. Allowed is any combination of FCW (fully
                conserved waters), HCW (half conserved waters) and WCW
                (weakly conserved waters). Defaults to ["FCW", "HCW", "WCW"].
        """
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
        self.normalize_orientations: bool = normalize_orientations
        self.plotend = plotend
        self.save_intermediate_results: bool = save_intermediate_results
        self.save_results_after_done: bool = save_results_after_done
        if self.plotend:
            if not (self.debugH < 2 or self.debugO < 2):
                self.plotend = False
                warnings.warn(
                    "plotend set to True while debugH or debugO are >1; setting back to False"
                )
        self.nsnaps: int = nsnaps
        self._waterO: list[np.ndarray] = []
        self._waterH1: list[np.ndarray] = []
        self._waterH2: list[np.ndarray] = []
        self._water_type: list[str] = []

    def save_clustering_options(
        self,
        clustering_type: str,
        clustering_algorithm: str,
        options: list[int, float],
        whichH: list[str],
        fname: str = "clust_options.dat",
    ) -> None:
        """Function that saves clustering options to a file.

        In order to restart the clustering procedure the intermediate
        results and clustering options need to be known. This function
        enables one to save the clustering options upon the start of
        clustering procedure. This happens automatically depending on
        ``save_intermediate_results`` parameter.

        Args:
            clustering_type (str): type of water clustering procedure to
                employ. The options are "multi_stage_reclustering" and
                "single_clustering".
            clustering_algorithm (str): Clustering algorith to use:
                "OPTICS" or "HDBSCAN"
            options (list): List containing additional options for each
                procedure type. If ``clustering_type`` is
                "single_clustering" options are ``[min_samp, xi]`` where
                xi can be any float if "HDBSCAN" is used. If
                ``clustering_type`` is "multi_stage_reclustering" the
                options are ``[lower_minsamp_pct, every_minsamp, xis]``.
                Again, in case of "HDBSCAN" xis can be a list with any
                single float.
            whichH (list[str]): which water types to seach for. Options
                are "onlyO" or any combination of "FCW", "HCW" and "WCW".
            fname (str, optional): file name to save clustering options
                to. Defaults to "clust_options.dat".
        """
        with open(fname, "w") as f:
            print(self.verbose, file=f)
            print(self.debugO, file=f)
            print(self.debugH, file=f)
            print(self.plotreach, file=f)
            print(self.plotend, file=f)
            print(self.numbpct_oxygen, file=f)
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
            print(self.normalize_orientations, file=f)
            print(self.nsnaps, file=f)
            print(clustering_type, file=f)
            print(clustering_algorithm, file=f)
            if type(options) != list:
                raise Exception("option has to be a list")
            for i in options:
                if type(i) != list and type(i) != np.ndarray:
                    print(i, file=f)
                elif type(i) == np.ndarray:
                    print(*list(i), file=f)
                else:
                    print(*i, file=f)
            print(*whichH, file=f)

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
        if not (H1 is None):
            H1 = np.delete(H1, elements, 0)
        if not (H2 is None):
            H2 = np.delete(H2, elements, 0)

        if self.save_intermediate_results:
            if not (H1 is None) and not (H2 is None):
                np.savetxt("water_coords_restart1.dat", np.c_[Odata, H1, H2])
            else:
                np.savetxt("water_coords_restart1.dat", np.c_[Odata])
            if os.path.isfile("water_coords_restart1.dat"):
                os.rename("water_coords_restart1.dat", "water_coords_restart.dat")
        return Odata, H1, H2

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
            self._waterO.append(i[0])
            if len(i) > 2:
                self._waterH1.append(i[1])
                self._waterH2.append(i[2])
            self._water_type.append(i[-1])

    def multi_stage_reclustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray,
        H2: np.ndarray,
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
        the beggining. Loops until no satisfactory clusterings are
        found. For more details see :cite:`conservedwatersearch2022`.

        Args:
            Odata (np.ndarray): Oxygen coordinates.
            H1 (np.ndarray): Hydrogen 1 orientations.
            H2 (np.ndarray): Hydrogen 2 orientations.
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
        if lower_minsamp_pct > 1.0000001 or lower_minsamp_pct < 0:
            raise Exception("lower_misamp_pct must be between 0 and 1")
        if type(every_minsamp) != int:
            raise Exception("every_minsamp must be integer")
        if every_minsamp <= 0 or every_minsamp > self.nsnaps:
            raise Exception("every_minsamp must be  0<every_minsamp<=nsnaps")
        for i in xis:
            if type(i) != float:
                raise Exception("xis must contain floats")
            if i > 1 or i < 0:
                raise Exception("xis should be between 0 and 1")
        for i in whichH:
            if not (i in ["FCW", "HCW", "WCW", "onlyO"]):
                raise Exception(
                    "whichH supports onlyO or any combination of FCW, HCW and WCW"
                )
        minsamps: list = list(
            reversed(
                range(
                    int(self.nsnaps * lower_minsamp_pct),
                    self.nsnaps + 1,
                    every_minsamp,
                )
            )
        )
        if clustering_algorithm == "HDBSCAN":
            lxis: list[float] = [0.0]
            allow_single = False
        elif clustering_algorithm == "OPTICS":
            lxis = xis
        else:
            raise Exception("clustering algorithm must be OPTICS or HDBSCAN")
        if self.save_intermediate_results:
            self.save_clustering_options(
                "multi_stage_reclustering",
                clustering_algorithm,
                [lower_minsamp_pct, every_minsamp, lxis],
                whichH,
            )
        found: bool = True
        if len(Odata) < self.nsnaps:
            found = False
        while found:
            found = False
            # loop over minsamps- from N(snapshots) to 0.75*N(snapshots)
            for i in minsamps:
                if clustering_algorithm == "OPTICS":
                    clust: OPTICS | HDBSCAN = OPTICS(min_samples=int(i), n_jobs=self.njobs)  # type: ignore
                    clust.fit(Odata)
                # loop over xi
                for j in lxis:
                    # recalculate reachability - OPTICS reachability has to be recaculated when changing minsamp
                    if clustering_algorithm == "HDBSCAN":
                        try:
                            import hdbscan
                        except:
                            raise Exception("install hdbscan")

                        clust = hdbscan.HDBSCAN(
                            min_cluster_size=int(self.nsnaps * self.numbpct_oxygen),
                            min_samples=int(i),
                            max_cluster_size=int(
                                self.nsnaps * (2 - self.numbpct_oxygen)
                            ),
                            cluster_selection_method="eom",
                            algorithm="best",
                            core_dist_n_jobs=self.njobs,
                            allow_single_cluster=allow_single,  # type: ignore
                        )
                        clust.fit(Odata)
                        clusters: np.ndarray = clust.labels_
                    elif clustering_algorithm == "OPTICS":
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
                        ff: Figure = __oxygen_clustering_plot(
                            Odata, clust, dbgt, self.debugO, self.plotreach
                        )
                    waters, idcs = self._analyze_oxygen_clustering(
                        Odata,
                        H1,
                        H2,
                        clusters,
                        stop_after_frist_water_found=True,
                        whichH=whichH,
                    )
                    if self.debugO == 1:
                        try:
                            import matplotlib.pyplot as plt
                        except:
                            raise Exception("install matplotlib")

                        plt.close(ff)
                    if len(waters) > 0:
                        found = True
                        if clustering_algorithm == "HDBSCAN" and allow_single:
                            allow_single = False
                        Odata, H1, H2 = self._delete_data(idcs, Odata, H1, H2)
                        self._add_water_solutions(waters)
                        if self.save_intermediate_results:
                            self.save_results(
                                fname="Clustering_results_temp1.dat",
                                typefname="Type_Clustering_results_temp1.dat",
                            )
                            if os.path.isfile(
                                "Clustering_results_temp1.dat"
                            ) and os.path.isfile("Type_Clustering_results_temp1.dat"):
                                os.rename(
                                    "Clustering_results_temp1.dat",
                                    "Clustering_results_temp.dat",
                                )
                                os.rename(
                                    "Type_Clustering_results_temp1.dat",
                                    "Type_Clustering_results_temp.dat",
                                )
                            else:
                                raise Warning(
                                    "unable to overwrite temp save files. Restarting might not work properly"
                                )
                        break
                if found:
                    break
            # check if size of remaining data set is bigger then number of snapshots
            if (
                clustering_algorithm == "HDBSCAN"
                and found is False
                and allow_single is False
            ):
                found = True
                allow_single = True
            if len(Odata) < self.nsnaps:
                found = False
        if (self.debugH == 1 or self.debugO == 1) and self.plotend:
            try:
                import matplotlib.pyplot as plt
            except:
                raise Exception("install matplotlib")

            plt.show()
        if self.save_results_after_done:
            self.save_results()

    def single_clustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray,
        H2: np.ndarray,
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
            H1 (np.ndarray): Hydrogen 1 orientations.
            H2 (np.ndarray): Hydrogen 2 orientations.
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
        if clustering_algorithm != "OPTICS" and clustering_algorithm != "HDBSCAN":
            raise Exception("clustering algorithm must be OPTICS or HDBSCAN")
        for i in whichH:
            if not (i in ["FCW", "HCW", "WCW", "onlyO"]):
                raise Exception(
                    "whichH supports onlyO or any combination of FCW, HCW and WCW"
                )
        if minsamp is None:
            minsamp = int(self.numbpct_oxygen * self.nsnaps)
        elif type(minsamp) != int:
            raise Exception("minsamp must be an int")
        elif minsamp > self.nsnaps or minsamp <= 0:
            raise Exception("minsamp must be between 0 and nsnaps")
        if xi is None:
            xi = 0.05
        elif type(xi) != float:
            raise Exception("xi must be a float")
        elif xi < 0 or xi > 1:
            raise Exception("xis should be between 0 and 1")
        if self.save_intermediate_results:
            self.save_clustering_options(
                "single_clustering",
                clustering_algorithm,
                [minsamp, xi],
                whichH,
            )
        if clustering_algorithm == "OPTICS":
            clust: OPTICS | HDBSCAN = OPTICS(
                min_samples=minsamp,
                min_cluster_size=int(self.numbpct_oxygen * self.nsnaps),
                xi=xi,
                n_jobs=self.njobs,
            )
            clust.fit(Odata)
            clusters = clust.labels_
        if clustering_algorithm == "HDBSCAN":
            try:
                import hdbscan
            except:
                raise Exception("install hdbscan")

            clust: OPTICS | HDBSCAN = hdbscan.HDBSCAN(
                min_cluster_size=int(self.nsnaps * self.numbpct_oxygen),
                min_samples=minsamp,
                max_cluster_size=int(self.nsnaps * (2 - self.numbpct_oxygen)),
                cluster_selection_method="eom",
                algorithm="best",
                core_dist_n_jobs=self.njobs,
            )
            clust.fit(Odata)
            clusters = clust.labels_
        # Debug stuff
        if self.debugO > 0:
            dbgt = ""
            if self.verbose > 0:
                (aa, bb) = np.unique(clusters, return_counts=True)
                dbgt = (
                    f"Oxygen clustering {type(clust)} minsamp={minsamp}, xi={xi}, {len(np.unique(clusters[clusters!=-1]))} clusters \n"
                    f"Required N(elem) range:{self.nsnaps*self.numbpct_oxygen:.2f} to {(2-self.numbpct_oxygen)*self.nsnaps}; (tar cls size={self.nsnaps} and numbpct={self.numbpct_oxygen:.2f})\n"
                    f"N(elements) for each cluster: {bb}\n"
                )
                print(dbgt)
            ff: Figure = __oxygen_clustering_plot(
                Odata, clust, dbgt, self.debugO, self.plotreach
            )
        waters, _ = self._analyze_oxygen_clustering(
            Odata,
            H1,
            H2,
            clusters,
            stop_after_frist_water_found=False,
            whichH=whichH,
        )
        if self.debugO == 1:
            try:
                import matplotlib.pyplot as plt
            except:
                raise Exception("install matplotlib")

            plt.close(ff)
        self._add_water_solutions(waters)
        if self.save_intermediate_results:
            self.save_results(
                fname="Clustering_results_temp1.dat",
                typefname="Type_Clustering_results_temp1.dat",
            )
            if os.path.isfile("Clustering_results_temp1.dat") and os.path.isfile(
                "Type_Clustering_results_temp1.dat"
            ):
                os.rename("Clustering_results_temp1.dat", "Clustering_results_temp.dat")
                os.rename(
                    "Type_Clustering_results_temp1.dat",
                    "Type_Clustering_results_temp.dat",
                )
        if (self.debugH == 1 or self.debugO == 1) and self.plotend:
            try:
                import matplotlib.pyplot as plt
            except:
                raise Exception("install matplotlib")

            plt.show()
        if self.save_results_after_done:
            self.save_results()

    def _analyze_oxygen_clustering(
        self,
        Odata: np.ndarray,
        H1: np.ndarray,
        H2: np.ndarray,
        clusters: np.ndarray,
        stop_after_frist_water_found: bool,
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
            H1 (np.ndarray): Hydrogen 1 orientations.
            H2 (np.ndarray): Hydrogen 2 orientations.
            clusters (np.ndarray):  Output of clustering
                results from OPTICS or HDBSCAN.
            stop_after_frist_water_found (bool): If True, the procedure
                is stopped after the first valid water is found and
                indicies of the oxygens and hydrogens belonging to this
                water are also returned.
            whichH (list[str]): Defines which water types to
                search for. Any combination of "FCW", "HWC" and "WCW" is
                allowed, or "onlyO" for oxygen clustering only.

        Returns:
            tuple[list[np.ndarray], list[int]]:
            returns two lists. First list contains valid conserved waters
            found. Each entry in the list is a list which contains the
            positions of oxygen, 2 hydrogen positions and water type
            found, the second list is a list of lists which contain
            arguments to be deleted if ``stop_after_frist_water_found``
            is True, else the second list is empty.
        """
        waters = []
        # Loop over all oxygen clusters (-1 is non cluster)
        for k in np.sort(np.unique(clusters[clusters != -1])):
            # Number of elements in oxygen cluster
            neioc: int = np.count_nonzero(clusters == k)
            # If number of elements in oxygen cluster is  Nsnap*0.85<Nelem<Nsnap*1.15 then ignore
            if (
                neioc < self.nsnaps * (2 - self.numbpct_oxygen)
                and neioc > self.nsnaps * self.numbpct_oxygen
            ):
                if self.verbose > 0:
                    print(f"O clust {k}, size {len(clusters[clusters==k])}\n")
                O_center = np.mean(Odata[clusters == k], axis=0)
                water = [O_center]
                if not (whichH[0] == "onlyO"):
                    # Construct array of hydrogen orientations
                    orientations = []
                    for i in np.argwhere(clusters == k):
                        orientations.append(H1[i])
                        orientations.append(H2[i])
                    orientations = np.asarray(orientations)[:, 0, :]
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
                        try:
                            import matplotlib.pyplot as plt
                        except:
                            raise Exception("install matplotlib")

                        plt.show()
                    if len(hyd) > 0:
                        # add water atoms for pymol visualisation
                        for i in hyd:
                            water.append(O_center + i[0])
                            water.append(O_center + i[1])
                            water.append(i[2])
                            waters.append(water)
                        # debug
                        if (
                            self.debugO == 1
                            and self.plotreach == 0
                            and self.debugH == 0
                        ):
                            try:
                                import matplotlib.pyplot as plt
                            except:
                                raise Exception("install matplotlib")

                            plt.show()
                        if stop_after_frist_water_found:
                            return waters, np.argwhere(clusters == k)
                else:
                    water.append("O_clust")
                    waters.append(water)
                    if stop_after_frist_water_found:
                        return waters, np.argwhere(clusters == k)
        return waters, []

    def restore_default_options(self, delete_results: bool = False) -> None:
        """This function restores all class options to defaults.

        Args:
            delete_results (bool, optional): If ``True`` all the results
                are overwritten with empty lists. Defaults to False.
        """
        self.numbpct_oxygen: float = 0.8
        self.normalize_orientations = True
        self.save_intermediate_results = True
        self.numbpct_hyd_orient_analysis: float = 0.85
        self.kmeans_ang_cutoff: float = 120
        self.kmeans_inertia_cutoff: float = 0.4
        self.conserved_angdiff_cutoff: float = 5
        self.conserved_angstd_cutoff: float = 17
        self.other_waters_hyd_minsamp_pct: float = 0.15
        self.noncon_angdiff_cutoff: float = 15
        self.halfcon_angstd_cutoff: float = 17
        self.weakly_angstd_cutoff: float = 20
        self.weakly_explained: float = 0.7
        self.xiFCW: list = [0.03]
        self.xiHCW: list = [0.05, 0.01]
        self.xiWCW: list = [0.05, 0.001]
        self.njobs: int = 1
        self.verbose: int = 0
        self.debugO: int = 0
        self.debugH: int = 0
        self.plotend: bool = False
        self.plotreach: bool = False
        if delete_results:
            self._water_type = []
            self._waterO = []
            self._waterH1 = []
            self._waterH2 = []

    def save_results(
        self,
        fname: str = "Clustering_results.dat",
        typefname: str = "Type_Clustering_results.dat",
    ) -> None:
        """Saves clustering results to files. Water coordinates are
        saved into the ``fname`` and water types are saved into the
        ``typefname`` file.

        Args:
            fname (str, optional): File name of the file that will
                contain saved water coordinates.
                Defaults to "Clustering_results.dat".
            typefname (str, optional): File name of the file that will
                contain water types.
                Defaults to "Type_Clustering_results.dat".
        """
        if len(self._waterH1) == 0 and len(self._waterH2) == 0:
            np.savetxt(fname, np.c_[self._waterO])
        else:
            np.savetxt(fname, np.c_[self._waterO, self._waterH1, self._waterH2])
        np.savetxt(typefname, np.c_[self._water_type], fmt="%s")

    def restart_cluster(
        self,
        options_file: str = "clust_options.dat",
        data_file: str = "water_coords_restart.dat",
        results_file: str = "Clustering_results_temp.dat",
        type_results_file: str = "Type_Clustering_results_temp.dat",
    ) -> None:
        """Read the clustering options and intermediate results from a
        file and restart the clustering procedure.

        Args:
            options_file (str, optional): File name of the options file.
                Defaults to "clust_options.dat".
            data_file (str, optional): File name of the file containing
                intermediate set of data of hydrogen and oxygen
                coordinates. Defaults to "water_coords_restart.dat".
            results_file (str, optional): File name containing partial
                results with determined water coordinates. Defaults to "Clustering_results_temp.dat".
            type_results_file (str, optional): File name containing
                partial results with water types.
                Defaults to "Type_Clustering_results_temp.dat".
        """
        if os.path.isfile(data_file):
            data: np.ndarray = np.loadtxt(fname=data_file)
            if len(data) == 3:
                Odata: np.ndarray = data
                H1: None | np.ndarray = None
                H2: None | np.ndarray = None
            elif len(data) == 9:
                Odata = data[:, :3]
                H1 = data[:, 3:6]
                H2 = data[:, 6:9]
        else:
            raise Exception("data file not found")
        if os.path.isfile(results_file):
            (
                self._water_type,
                self._waterO,
                self._waterH1,
                self._waterH2,
            ) = read_results(results_file, type_results_file)
        else:
            raise Exception("results file not found")
        self.read_class_options(options_file=options_file)
        (
            clustering_type,
            clustering_algorithm,
            options,
            whichH,
        ) = self.read_water_clust_options(options_file=options_file)
        if clustering_type == "multi_stage_reclustering":
            self.multi_stage_reclustering(
                Odata,  # type: ignore
                H1,  # type: ignore
                H2,  # type: ignore
                clustering_algorithm,
                options[0],
                options[1],  # type: ignore
                options[2],  # type: ignore
                whichH,
            )
        elif clustering_type == "single_clustering":
            self.single_clustering(
                Odata,  # type: ignore
                H1,  # type: ignore
                H2,  # type: ignore
                clustering_algorithm,
                options[0],  # type: ignore
                options[1],
                whichH,
            )
        else:
            raise Exception("incompatible clustering type")

    def read_water_clust_options(
        self, options_file="clust_options.dat"
    ) -> tuple[str, str, list[int | float], list[str]]:
        """Reads clustering procedure type and parameters and returns them.

        Args:
            options_file (str, optional): File to read clustering
                procedure parameters from.
                Defaults to "clust_options.dat".

        Returns:
            Tuple[str, str, list[int | float], list[str]]:
            Returns clustering procedure type, clustering algorithm, options
            for the procedure type and which water types to determine
        """
        if os.path.isfile(options_file):
            f: TextIOWrapper = open(options_file, "r")
            lines: list[str] = f.read().splitlines()
            clustering_type: str = lines[21].strip(" ")
            if clustering_type == "multi_stage_reclustering":
                clustering_algorithm: str = lines[22].strip(" ")
                lower_minsamp_pct: float = float(lines[23])
                every_minsamp: int = int(lines[24])
                xis: list[float] = [float(i) for i in lines[25].split(" ")]
                whichH: list[str] = [i for i in lines[26].split(" ")]
                options = [lower_minsamp_pct, every_minsamp, xis]
            elif clustering_type == "single_clustering":
                clustering_algorithm = lines[22].strip(" ")
                minsamp: float = float(lines[23])
                xi: float = float(lines[24])
                whichH = [i for i in lines[25].split(" ")]
                options = [minsamp, xi]
            else:
                raise Exception("incompatible clustering type")
            return clustering_type, clustering_algorithm, options, whichH
        else:
            raise Exception("options file not found")

    def read_class_options(self, options_file="clust_options.dat") -> None:
        """Reads all class clustering options from save file and sets
        the parameters. Reads all parameters except clustering protocol
        and protocol parameters.

        Args:
            options_file (str, optional): File name of the options file.
                Defaults to "clust_options.dat".
        """
        if os.path.isfile(options_file):
            f: TextIOWrapper = open(options_file, "r")
            lines: list[str] = f.read().splitlines()
            self.verbose = int(lines[0])
            self.debugO = int(lines[1])
            self.debugH = int(lines[2])
            self.plotreach = lines[3] == "True"
            self.plotend = lines[4] == "True"
            self.numbpct_oxygen = float(lines[5])
            self.numbpct_hyd_orient_analysis = float(lines[6])
            self.kmeans_ang_cutoff = float(lines[7])
            self.kmeans_inertia_cutoff = float(lines[8])
            self.conserved_angdiff_cutoff = float(lines[9])
            self.conserved_angstd_cutoff = float(lines[10])
            self.other_waters_hyd_minsamp_pct = float(lines[11])
            self.noncon_angdiff_cutoff = float(lines[12])
            self.halfcon_angstd_cutoff = float(lines[13])
            self.weakly_angstd_cutoff = float(lines[14])
            self.weakly_explained = float(lines[15])
            self.xiFCW = [float(i) for i in lines[16].split(" ")]
            self.xiHCW = [float(i) for i in lines[17].split(" ")]
            self.xiWCW = [float(i) for i in lines[18].split(" ")]
            self.normalize_orientations = lines[19] == "True"
            self.nsnaps = int(lines[20])
        else:
            raise Exception("options file not found")

    @classmethod
    def create_from_file(
        cls,
        options_file: str = "clust_options.dat",
    ) -> WaterClustering:
        """Create a WaterClustering class from saved clustering options
        file.

        Args:
            options_file (str, optional): File name containig saved
                clustering options. Defaults to "clust_options.dat".

        Returns:
            WaterClustering: creates an instance of :py:class:`WaterClustering`
            class by reading options from a file.
        """
        cls = cls(0)
        if os.path.isfile(options_file):
            f: TextIOWrapper = open(options_file, "r")
            lines: list[str] = f.read().splitlines()
            cls.verbose = int(lines[0])
            cls.debugO = int(lines[1])
            cls.debugH = int(lines[2])
            cls.plotreach = lines[3] == "True"
            cls.plotend = lines[4] == "True"
            cls.numbpct_oxygen = float(lines[5])
            cls.numbpct_hyd_orient_analysis = float(lines[6])
            cls.kmeans_ang_cutoff = float(lines[7])
            cls.kmeans_inertia_cutoff = float(lines[8])
            cls.conserved_angdiff_cutoff = float(lines[9])
            cls.conserved_angstd_cutoff = float(lines[10])
            cls.other_waters_hyd_minsamp_pct = float(lines[11])
            cls.noncon_angdiff_cutoff = float(lines[12])
            cls.halfcon_angstd_cutoff = float(lines[13])
            cls.weakly_angstd_cutoff = float(lines[14])
            cls.weakly_explained = float(lines[15])
            cls.xiFCW = [float(i) for i in lines[16].split(" ")]
            cls.xiHCW = [float(i) for i in lines[17].split(" ")]
            cls.xiWCW = [float(i) for i in lines[18].split(" ")]
            cls.normalize_orientations = lines[19] == "True"
            cls.nsnaps = int(lines[20])
        else:
            raise Exception("options file not found")
        return cls

    def visualise_pymol(
        self,
        aligned_protein: str = "aligned.pdb",
        output_file: str | None = None,
        active_site_ids: list[int] | None = None,
        crystal_waters: str | None = None,
        ligand_resname: str | None = None,
        dist: float = 10.0,
        density_map: str | None = None,
    ) -> None:
        """Visualise results using `pymol <https://pymol.org/>`__.

        Args:
            aligned_protein (str, optional): file name containing protein
                configuration trajectory was aligned to. Defaults to "aligned.pdb".
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
        )

    def visualise_nglview(
        self,
        aligned_protein: str = "aligned.pdb",
        active_site_ids: list[int] | None = None,
        crystal_waters: str | None = None,
        density_map: str | None = None,
    ) -> NGLWidget:
        """Visualise the results using `nglview <https://github.com/nglviewer/nglview>`__.

        Args:
            aligned_protein (str, optional): File containing protein
                configuration the original trajectory was aligned to.
                Defaults to "aligned.pdb".
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


def __oxygen_clustering_plot(
    Odata: np.ndarray,
    cc: OPTICS,
    title: str,
    debugO: int,
    plotreach: bool = False,
) -> Figure:
    """Function for plotting oxygen clustering results.

    For debuging oxygen clustering. Not ment for general usage.

    """
    if type(cc) != OPTICS:
        plotreach = False
    if debugO > 0:
        try:
            import matplotlib.pyplot as plt
        except:
            raise Exception("install matplotlib")

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
            try:
                import matplotlib.pyplot as plt
            except:
                raise Exception("install matplotlib")

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
        try:
            import matplotlib.pyplot as plt
        except:
            raise Exception("install matplotlib")

        plt.show()
    return fig
