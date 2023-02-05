from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import numpy as np
from sklearn.cluster import OPTICS, KMeans


def hydrogen_orientation_analysis(
    orientations: np.ndarray,
    pct_size_buffer: float = 0.85,
    kmeans_ang_cutoff: float = 120,
    kmeans_inertia_cutoff: float = 0.4,
    FCW_angdiff_cutoff: float = 5,
    FCW_angstd_cutoff: float = 17,
    min_samp_data_size_pct: float = 0.15,
    nonFCW_angdiff_cutoff: float = 15,
    HCW_angstd_cutoff: float = 17,
    WCW_angstd_cutoff: float = 20,
    weakly_explained: float = 0.7,
    xiFCW: list[float] = [0.03],
    xiHCW: list[float] = [0.05, 0.01],
    xiWCW: list[float] = [0.05, 0.001],
    njobs: int = 1,
    verbose: int = 0,
    debugH: int = 0,
    plotreach: bool = False,
    which: list[str] = ["FCW", "HCW", "WCW"],
    normalize_orientations: bool = True,
) -> list:
    """Determines if the water cluster is conserved and of what type.

    High level function that does hydrogen orientation analysis. Checks
    if the water cluster belongs into one of the following groups by
    analizing hydrogen orientations:

    - FCW (Fully Conserved Water): hydrogens are strongly oriented in
      two directions with angle of 104.5
    - HCW (Half Conserved Water): one set (cluster) of hydrogens is
      oriented in certain directions and other are spread into different
      orientations with angle of 104.5
    - WCW (Weakly Conserved Water): several orientation combinations
      exsist with satisfying water angles

    See :cite:`conservedwatersearch2022` for more information on water
    classification :ref:`conservedwaters:theory, background and methods`.
    If orientations don't satisfy the criteria for any of the waters, an
    empty list is returned.

    Args:
        orientations (np.ndarray): array of hydrogen
            orientations in space
        pct_size_buffer (float, optional): Minimum allowed size of the
            hydrogen orientation cluster. Defaults to 0.85.
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
        min_samp_data_size_pct (float, optional): Minimum samples to
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
        xiFCW (list, optional): Xi value for OPTICS clustering for FCW. Don't
            touch this unless you know what you are doing. Defaults to [0.03].
        xiHCW (list, optional): Xi value for OPTICS clustering for HCW. Don't
            touch this unless you know what you are doing.
            Defaults to [0.05, 0.01].
        xiWCW (list, optional): Xi value for OPTICS clustering for WCW. Don't
            touch this unless you know what you are doing.
            Defaults to [0.05, 0.001].
        njobs (int, optional): how many cpu cores to use for clustering.
            Defaults to 1.
        verbose (int, optional): verbosity of output. Defaults to 0.
        debugH (int, optional): debug level for orientations. Defaults to 0.
        plotreach (bool, optional): weather to plot the reachability
            plot for OPTICS when debuging. Defaults to False.
        which (list, optional): list of strings denoting which water types to
            search for. Allowed is any combination of FCW (fully
            conserved waters), HCW (half conserved waters) and WCW
            (weakly conserved waters). Defaults to ["FCW", "HCW",
            "WCW"].
        normalize_orientations (bool, optional): weather to normalize
            the orientation vectors to unit distance. Defaults to True.

    Returns:
        list: returns a list containing two orientations of hydrogens
        and water classification string ("FCW", "HCW", "WCW"), if
        not conserved returns an empty list
    """
    orientations = np.array(orientations)
    # check length of orientations - it has to be bigger then 1 and even
    orishape = orientations.shape
    if len(orishape) != 2:
        raise ValueError("Orientations have to be a 2D array")
    Hnum = orishape[0]
    dimnum = orishape[1]
    if dimnum != 3:
        raise ValueError("Orientations must be vectors of dimension 3")
    if Hnum < 2 or Hnum % 2 != 0:
        raise ValueError(
            "Number of orientations must be even! Each water molecule has 2 hydrogen atoms!"
        )
    if normalize_orientations:
        orientations = orientations / np.linalg.norm(
            orientations, axis=1, keepdims=True
        )
    # For Debug
    np.set_printoptions(precision=4)
    # Check if water is conserved
    if "FCW" in which:
        for xi in xiFCW:
            conserved = find_fully_conserved_orientations(
                orientations,
                pct_size_buffer=pct_size_buffer,
                kmeans_ang_cutoff=kmeans_ang_cutoff,
                kmeans_inertia_cutoff=kmeans_inertia_cutoff,
                angdiff_cutoff=FCW_angdiff_cutoff,
                angstd_cutoff=FCW_angstd_cutoff,
                xi=xi,
                njobs=njobs,
                verbose=verbose,
                debugH=debugH,
                plotreach=plotreach,
            )
            if len(conserved) > 0:
                return conserved
    if "HCW" in which:
        for xi in xiHCW:
            half_conserved = find_half_conserved_orientations(
                orientations,
                pct_size_buffer=pct_size_buffer,
                min_samp_data_size_pct=min_samp_data_size_pct,
                angdiff_cutoff=nonFCW_angdiff_cutoff,
                angstd_cutoff=HCW_angstd_cutoff,
                xi=xi,
                njobs=njobs,
                verbose=verbose,
                debugH=debugH,
                plotreach=plotreach,
            )
            if len(half_conserved) > 0:
                return half_conserved
    if "WCW" in which:
        for xi in xiWCW:
            semi_conserved = find_weakly_conserved_orientations(
                orientations,
                pct_size_buffer=pct_size_buffer,
                min_samp_data_size_pct=min_samp_data_size_pct,
                pct_explained=weakly_explained,
                angdiff_cutoff=nonFCW_angdiff_cutoff,
                angstd_cutoff=WCW_angstd_cutoff,
                xi=xi,
                njobs=njobs,
                verbose=verbose,
                debugH=debugH,
                plotreach=plotreach,
            )
            if len(semi_conserved):
                return semi_conserved
    return []
    # I want this to return an array with hydrogen orientations combinations


def find_fully_conserved_orientations(
    orientations: np.ndarray,
    pct_size_buffer: float = 0.85,
    kmeans_ang_cutoff: float = 120,
    kmeans_inertia_cutoff: float = 0.4,
    angdiff_cutoff: float = 5,
    angstd_cutoff: float = 17.0,
    xi: float = 0.03,
    njobs: int = 1,
    verbose: int = 0,
    debugH: int = 0,
    plotreach: bool = False,
) -> list:
    """Check if orientations belong to FCW.

    Checks if given oxygen cluster can be considered as a fully
    conserved water based on hydrogen orientations. Fully conserved
    water is one which has well defined hydrogen orientations in two
    distinctive groups (ie strongly hydrogen bonded for both hydrogens).
    To check if water is conserved, one first checks if k means
    clustering of hydrogen orientations gives two destinctive clusters
    with low inertia and required angle between the clusters. Afterwards
    more rigorous check is carried out with OPTICS clustering where
    again the sperad of orientations and angle is considered.

    Args:
        orientations (np.ndarray): array of hydrogen
            orientations in space
        pct_size_buffer (float, optional): Minimum allowed size of the
            hydrogen orientation cluster. Defaults to 0.85.
        kmeans_ang_cutoff (float, optional): Maximum value of angle (in
            deg) allowed for FCW in kmeans clustering to be considered
            correct water angle. Defaults to 120.
        kmeans_inertia_cutoff (float, optional): upper limit allowed on
            kmeans inertia (measure of spread of data in a cluster).
            Defaults to 0.4.
        angdiff_cutoff (float, optional): Maximum value of angle (in
            deg) allowed for FCW in OPTICS/HDBSCAN clustering to be
            considered correct water angle. Defaults to 5.
        angstd_cutoff (float, optional): Maximal standard deviation
            of angle distribution of orientations of two hydrogens
            allowed for water to be considered FCW. Defaults to 17.
        xi (float, optional): Xi value for OPTICS clustering for FCW. Don't
            touch this unless you know what you are doing.
            Defaults to 0.03.
        njobs (int, optional): how many cpu cores to use for clustering.
            Defaults to 1.
        verbose (int, optional): verbosity of output. Defaults to 0.
        debugH (int, optional): debug level for orientations. Defaults to 0.
        plotreach (bool, optional): weather to plot the reachability
            plot for OPTICS when debuging. Defaults to False.

    Returns:
        list: returns a list containing two orientations of hydrogens
        and water classification string "FCW", if not FCW returns
        empty list
    """
    # number of elements in oxygen cluster
    neioc = int(len(orientations) / 2)
    # for debugging
    counts = np.asarray([0])
    values = np.asarray([0])
    angs12 = 0
    angs21 = 0
    biggest = 0
    secondbiggest = 0
    labels = 0
    cc = 0
    ss = ""
    # conserved hydrogens
    fully_conserved = []
    # Kmeans clustering prepareation - for conserved water analysis only
    kmeans = KMeans(n_clusters=2)
    # fit kmeans clustering
    kmeans.fit(orientations)
    if len(kmeans.cluster_centers_) == 2:
        Kcv1 = kmeans.cluster_centers_[0]
        Kcv2 = kmeans.cluster_centers_[1]
        # Calculate angles between centers of clusters that represent hydrogen orientation
        ang = (
            360.0
            / (2.0 * np.pi)
            * np.arccos(
                np.clip(
                    np.dot(
                        Kcv1 / np.linalg.norm(Kcv1),
                        Kcv2 / np.linalg.norm(Kcv2),
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        # check if angle between cluster centres is about right and check if spread (variance) of orientations is sufficiently low
        if (
            ang < kmeans_ang_cutoff
            and kmeans.inertia_ / len(orientations) < kmeans_inertia_cutoff
        ):
            # Perform OPTICS clustering on hydrogen orientations; minsamp is same for all water types, however here we force min_cluster size to be given.
            msp = int(neioc * pct_size_buffer)
            msp = msp if msp >= 1 else 1
            cc = OPTICS(min_samples=msp, xi=xi, n_jobs=njobs)
            cc.fit(orientations)
            labels = cc.labels_
            # Calculate number of elements in each cluster
            (values, counts) = np.unique(labels[labels != -1], return_counts=True)
            # if number of optics clusters for hydrogen clustering is >0
            if len(np.sort(np.unique(labels[labels != -1]))) == 2:
                # find two biggest clusters
                biggest = values[np.argsort(counts)[-1]]
                secondbiggest = values[np.argsort(counts)[-2]]
                # check if size of two biggest clusters are between 0.85 and 1.15*sizeofOxygenCluster
                if (
                    len(orientations[labels == biggest])
                    <= neioc * (2 - pct_size_buffer)
                    and len(orientations[labels == secondbiggest])
                    <= neioc * (2 - pct_size_buffer)
                    and len(orientations[labels == biggest]) > neioc * pct_size_buffer
                    and len(orientations[labels == secondbiggest])
                    > neioc * pct_size_buffer
                ):
                    # calculate the average orientation of two biggest clusters
                    avang1 = np.mean(orientations[labels == biggest], axis=0)
                    avang2 = np.mean(orientations[labels == secondbiggest], axis=0)
                    # calculate average angle between average orientation of one cluster with elements(orientations) of the other cluster
                    angs12 = (
                        np.arccos(
                            np.dot(orientations[labels == secondbiggest], avang1.T)
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    angs21 = (
                        np.arccos(np.dot(orientations[labels == biggest], avang2.T))
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    # check if average angles are low and that their std. deviation is low
                    if (
                        np.abs(np.mean(angs21) - np.mean(angs12)) < angdiff_cutoff
                        and np.std(angs12) < angstd_cutoff
                        and np.std(angs21) < angstd_cutoff
                    ):
                        # conserved
                        if verbose > 0:
                            print("conserved found")
                        fully_conserved.append(
                            __return_normalized_orientation_pair(
                                orientations,
                                labels,
                                biggest,
                                secondbiggest,
                                "FCW",
                            )
                        )
    # Debug needs neoic,orientations,kmeans,ang,k,counts,biggest,secondbiggest,angs12,angs21,cc
    # kmeans
    if verbose > 0 or debugH > 0:
        sk = (
            f"Conserved - kmeans analysis:\n"
            f"angle - mean: {ang:.2f}(calced)<{kmeans_ang_cutoff:.2f}; inertia per H: {kmeans.inertia_/len(orientations):.2f}(calced)<{kmeans_inertia_cutoff:.2f}\n"
        )
    if debugH == 2 or (debugH == 1 and len(fully_conserved) > 0):
        __plot3Dorients(111, kmeans.labels_, orientations, sk)
    # OPTICS
    if debugH > 0 or verbose > 0:
        ss = (
            f"Fully Conserved - OPTICS analysis;depends on pct size buffer:{pct_size_buffer:.2f}\n"
            f"cluster sizes: {counts}; clust labels:{values};biggest: {biggest}; secondbiggest: {secondbiggest}\n"
            f"clusters size: {neioc*pct_size_buffer:.2f}<{len(orientations[labels == biggest])},{len(orientations[labels == secondbiggest])}(calced)<{neioc*(2-pct_size_buffer):.2f}\n"
            f"angle mean: {np.abs(np.mean(angs21)-np.mean(angs12)):.2f}(calced)<{angdiff_cutoff:.2f}); 12 std : {np.std(angs12):.2f};21 std: {np.std(angs21):.2f} (both <{angstd_cutoff:.2f})\n"
        )
    # Printing
    if verbose == 2 or (verbose == 1 and len(fully_conserved) > 0):
        print(sk + ss)
    # Debug plots
    __hydrogen_orient_plots(
        labels,
        orientations,
        cc,
        ss,
        f"Reachability of optics plot\n minsamples={int(neioc*pct_size_buffer)}; xi={xi}",
        len(fully_conserved) > 0,
        debugH,
        plotreach,
    )
    # End of function
    return fully_conserved


def find_half_conserved_orientations(
    orientations: np.ndarray,
    pct_size_buffer: float = 0.85,
    min_samp_data_size_pct: float = 0.35,
    angdiff_cutoff: float = 15,
    angstd_cutoff: float = 17.0,
    xi: float = 0.01,
    njobs: int = 1,
    verbose: int = 0,
    debugH: int = 0,
    plotreach: bool = False,
) -> list:
    """Checks if given orientations belong to HCW.

    Checks if given oxygen cluster can be considered as a half conserved
    water based on hydrogen orientations. Half conserved water is one
    which has one well defined hydrogen orientation (ie one strongly
    hydrogen bonded hydrogen). To check if water is half conserved, one
    calculates OPTICS clustering of hydrogen orientations. One then
    loops over clusters in an atempt to find a hydrogen orientation
    cluster which is the size of oxygen cluster and weather the angle
    between that cluster with all other orientations is of right angle
    and if spread of orientations is sufficiently low.

    Args:
        orientations (np.ndarray): array of hydrogen
            orientations in space
        pct_size_buffer (float, optional): Minimum allowed size of the
            hydrogen orientation cluster. Defaults to 0.85.
        min_samp_data_size_pct (float, optional): Minimum samples to
            choose for OPTICS or HDBSCAN clustering as percentage of
            number of water molecules considered for HCW and WCW.
            Defaults to 0.15.
        angdiff_cutoff (float, optional): Maximum standard
            deviation of angle allowed for HCW to be considered
            correct water angle. Defaults to 15.
        HCW_angstd_cutoff (float, optional): Maximum standard deviation
            cutoff for WCW angles to be considered correct water angles.
            Defaults to 17.
        xi (float, optional): Xi value for OPTICS clustering for HCW. Don't
            touch this unless you know what you are doing.
            Defaults to 0.01.
        njobs (int, optional): how many cpu cores to use for clustering.
            Defaults to 1.
        verbose (int, optional): verbosity of output. Defaults to 0.
        debugH (int, optional): debug level for orientations. Defaults to 0.
        plotreach (bool, optional): weather to plot the reachability
            plot for OPTICS when debuging. Defaults to False.

    Returns:
        list: returns a list containing two orientations of hydrogens
        and water classification string "HCW", if not HCW returns
        an empty list
    """
    # number of elements in oxygen cluster
    neioc = int(len(orientations) / 2)
    half_conserved = []
    sk = ""
    ss = ""
    # Optics clustering for hydrogen orientations - minsamp is same for all water types
    msp = int(neioc * min_samp_data_size_pct)
    msp = msp if msp >= 1 else 1
    cc = OPTICS(
        min_samples=msp,
        xi=xi,
        n_jobs=njobs,
    )
    cc.fit(orientations)
    # lebels without -1
    labels = cc.labels_[cc.labels_ != -1]
    (values, counts) = np.unique(labels, return_counts=True)
    ind = np.argmax(counts[counts != -1])
    bestcand = values[ind]
    labels = cc.labels_
    # if there is more then 1 cluster
    if len(np.unique(labels)) > 1:
        N_elems = len(orientations[labels == bestcand])
        # Debug
        if verbose > 0:
            sk += f"Cluster {bestcand} has {neioc*pct_size_buffer:.2f}<{N_elems}<{neioc*(2-pct_size_buffer):.2f} elements\n"
        # check if number of elements in hydrogen orientation cluster is between 0.80 and 1.20 elements of Oxygen cluster
        if (
            N_elems < neioc * (2 - pct_size_buffer)
            and N_elems > neioc * pct_size_buffer
        ):
            # calculating average angle between average orientation of selected cluster and all other orientations
            avang1 = np.mean(orientations[labels == bestcand], axis=0)
            angs1all = (
                np.arccos(np.dot(orientations[labels != bestcand], avang1.T))
                * 360.0
                / (2.0 * np.pi)
            )
            # Debug
            if verbose > 0 or debugH > 0:
                sk += f"result angles: mean= {np.abs(np.mean(angs1all)-104.5):.2f}(Calced)<{angdiff_cutoff:.2f}; std dev={np.std(angs1all):.2f}(Calced)<{angstd_cutoff:.2f}\n"
            # check if angle and its std deviation is satisfactory
            if (
                np.abs(np.mean(angs1all) - 104.5) < angdiff_cutoff
                and np.std(angs1all) < angstd_cutoff
            ):
                for j in np.unique(labels[labels != bestcand]):
                    # -1 has to be inclueded in case main cluster is ok and -1 is everywhere, to produce a HCW still
                    if verbose > 0:
                        print("half conserved found", bestcand, j)
                    half_conserved.append(
                        __return_normalized_orientation_pair(
                            orientations, labels, bestcand, j, "HCW"
                        )
                    )
    # Debug
    if verbose > 0 or debugH > 0:
        ss = (
            f"Half Conserved - OPTICS analysis;xi={xi}"
            f"best: {bestcand}; N_hyd_cls (w/o -1): {len(np.unique(labels[labels!=-1]))};\n N elem in biggest:{neioc*(2-pct_size_buffer):.2f}>{len(orientations[labels==bestcand])}>{neioc*pct_size_buffer:.2f}\n"
        )
    # Printing
    if verbose == 2 or (verbose == 1 and len(half_conserved) > 0):
        print(ss + sk)
    # Debug plots
    __hydrogen_orient_plots(
        labels,
        orientations,
        cc,
        ss + sk,
        f"Reachability of optics plot\n minsamples={int(neioc*min_samp_data_size_pct)}; xi={xi}",
        len(half_conserved) > 0,
        debugH,
        plotreach,
    )
    # End of function
    return half_conserved


def find_weakly_conserved_orientations(
    orientations: np.ndarray,
    pct_size_buffer: float = 0.85,
    lower_bound_pct_buffer: float = 0.35,
    min_samp_data_size_pct: float = 0.15,
    pct_explained: float = 0.7,
    angdiff_cutoff: float = 15,
    angstd_cutoff: float = 20.0,
    xi: float = 0.01,
    njobs: int = 1,
    verbose: int = 0,
    debugH: int = 0,
    plotreach: bool = False,
) -> list:
    """Checks if given orientations belong to WCW.

    Checks if given oxygen cluster can be considered as a weakly
    conserved water based on hydrogen orientations. weakly conserved
    water is one which has no well defined hydrogen orientation (ie no
    strongly hydrogen bonded hydrogen) but still has distinct hydrogen
    orientational clusters. To check if water is weakly conserved, one
    calculates OPTICS clustering of hydrogen orientations. One then
    loops over clusters in an atempt to find a pair of hydrogen
    orientation clusters which is of the same size and weather the angle
    between the two clusters is of right angle and if spread of
    orientations is sufficiently low. Aditionally triplets are checked
    as well. Here we do the same check but we are looking at cluster one
    vs two other clusters combined.

    Args:
        orientations (np.ndarray): array of hydrogen
            orientations in space
        pct_size_buffer (float, optional): Minimum allowed size of the
            hydrogen orientation cluster. Defaults to 0.85.
        min_samp_data_size_pct (float, optional): Minimum samples to
            choose for OPTICS or HDBSCAN clustering as percentage of
            number of water molecules considered for HCW and WCW.
            Defaults to 0.15.
        pct_explained (float, optional): percentage of explained
            hydrogen orientations for water to be considered WCW.
            Defaults to 0.7.
        angdiff_cutoff (float, optional): Maximum standard
            deviation of angle allowed for WCW to be considered
            correct water angle. Defaults to 15.
        angstd_cutoff (float, optional): Maximum standard deviation
            cutoff for WCW angles to be considered correct water angles.
            Defaults to 20.
        xi (float, optional): Xi value for OPTICS clustering for WCW. Don't
            touch this unless you know what you are doing.
            Defaults to 0.01.
        njobs (int, optional): how many cpu cores to use for clustering.
            Defaults to 1.
        verbose (int, optional): verbosity of output. Defaults to 0.
        debugH (int, optional): debug level for orientations. Defaults to 0.
        plotreach (bool, optional): weather to plot the reachability
            plot for OPTICS when debuging. Defaults to False.

    Returns:
        list: returns a list containing two orientations of hydrogens
        and water classification string "WCW", if not WCW returns
        an empty list
    """
    # number of elements in oxygen cluster
    neioc = int(len(orientations) / 2)
    # Optics clustering for hydrogen orientations - minsamp is same for all water types
    msp = int(neioc * min_samp_data_size_pct)
    msp = msp if msp >= 1 else 1
    cc = OPTICS(
        min_samples=msp,
        xi=xi,
        n_jobs=njobs,
    )
    cc.fit(orientations)
    weakly_conserved = []
    # lebels without -1
    labels = cc.labels_
    (_, counts) = np.unique(labels, return_counts=True)
    weakly = []
    sk = ""
    ss = ""
    # array of already used clusters
    used = []
    # list of labels
    lbls = np.unique(labels)
    # if there is more then 1 cluster
    if len(np.unique(labels)) > 1:
        # Chekc over pairs of clusters
        # loop over OPTICS clusters
        for ii in lbls[lbls != -1]:
            # check if in used
            if ii in used:
                continue
            N_elems = len(orientations[labels == ii])
            # Debug
            if verbose > 0 or debugH > 0:
                sk += f"Cluster {ii} has {neioc*lower_bound_pct_buffer:.2f}<{N_elems}<{neioc*(2-pct_size_buffer):.2f} elements\n"
            # check if size of hydorgen orientation cluster is between 1.20 and lower_bound_pct_buffer times number of elements in oxygen cluster
            if (
                N_elems < neioc * (2 - pct_size_buffer)
                and N_elems > neioc * lower_bound_pct_buffer
            ):
                avang1 = np.mean(orientations[labels == ii], axis=0)
                # loop over clusters but not ii
                lblsni = lbls[lbls != ii]
                for jj in lblsni:
                    # check if already used
                    if ii in used:
                        break
                    if jj in used:
                        continue
                    # calculate average angle between average orientation of ii and all orientations in cluster jj
                    angs1j = (
                        np.arccos(np.dot(orientations[labels == jj], avang1.T))
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    N_elems_jj = len(orientations[labels == jj])
                    # Debug
                    if verbose > 0 or debugH > 0:
                        sk += f"cluster combo:{ii} & {jj}size:{N_elems},{neioc*lower_bound_pct_buffer:.2f}<{N_elems_jj}<{neioc*(2-pct_size_buffer):.2f}\n"
                        sk += f"size comparison {np.abs(N_elems -N_elems_jj)} (calced) < {(1 - pct_size_buffer) * np.max([N_elems, N_elems_jj])} \n"
                        sk += f"ang diff={np.abs(np.mean(angs1j)-104.5):.2f}(calced)<{angdiff_cutoff:.2f},std dev:{angstd_cutoff:.2f}>{np.std(angs1j):.2f}(calced)\n"
                    # check if size of new cluster and check if size of clusters is about equal
                    if (
                        N_elems_jj > neioc * (2 - pct_size_buffer)
                        or N_elems_jj < neioc * lower_bound_pct_buffer
                        or np.abs(N_elems - N_elems_jj)
                        > (1 - pct_size_buffer) * np.max([N_elems, N_elems_jj])
                    ):
                        continue
                    # check if angles and std devs are good
                    if (
                        np.abs(np.mean(angs1j) - 104.5) < angdiff_cutoff
                        and np.std(angs1j) < angstd_cutoff
                        and len(orientations[labels == jj])
                        > neioc * lower_bound_pct_buffer
                        and len(orientations[labels == jj])
                        < neioc * (2 - pct_size_buffer)
                    ):
                        used.append(ii)
                        used.append(jj)
                        weakly.append([ii, jj])
                        break
        # check for triplets if cluster ii has same size as clusters jj+kk and satisfactory angles
        # loop over OPTICS clusters
        for ii in lbls[lbls != -1]:
            # check if in used
            if ii in used:
                continue
            N_elems = len(orientations[labels == ii])
            # average orientation of ii
            avang1 = np.mean(orientations[labels == ii], axis=0)
            # loop over clusters but not ii
            lblsni = lbls[lbls != ii]
            for jj in lblsni:
                # check if already used
                if ii in used:
                    break
                if jj in used:
                    continue
                N_elems_jj = len(orientations[labels == jj])
                avangj = np.mean(orientations[labels == jj], axis=0)
                # loop over clusters but not ii or jj
                lblsnij = lblsni[lblsni != jj]
                for kk in lblsnij:
                    if ii in used:
                        break
                    if jj in used:
                        break
                    if kk in used:
                        continue
                    N_elems_kk = len(orientations[labels == kk])
                    avangk = np.mean(orientations[labels == kk], axis=0)
                    # calculate average angle between average orientation of ii and all orientations in cluster jj and kk
                    angs1jk = (
                        np.arccos(
                            np.dot(
                                orientations[(labels == jj) | (labels == kk)],
                                avang1.T,
                            )
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    angs1j = (
                        np.arccos(
                            np.dot(
                                orientations[(labels == jj)],
                                avang1.T,
                            )
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    angs1k = (
                        np.arccos(
                            np.dot(
                                orientations[(labels == kk)],
                                avang1.T,
                            )
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    angsjk = (
                        np.arccos(
                            np.dot(
                                orientations[(labels == kk)],
                                avangj.T,
                            )
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    angskj = (
                        np.arccos(
                            np.dot(
                                orientations[(labels == jj)],
                                avangk.T,
                            )
                        )
                        * 360.0
                        / (2.0 * np.pi)
                    )
                    # Debug
                    if verbose > 0 or debugH > 0:
                        sk += f"cluster combo:{ii} & {jj} & {kk} size:{N_elems},{N_elems_jj}, {N_elems_kk} \n"
                        sk += f"size check {np.abs(N_elems - (N_elems_jj + N_elems_kk))} (calced)< {(1 - pct_size_buffer) * np.max([N_elems, N_elems_jj + N_elems_kk])}\n"
                        sk += f"ang diff={np.abs(np.mean(angs1jk)-104.5):.2f}(calced)<{angdiff_cutoff:.2f},std dev:{angstd_cutoff:.2f}>{np.std(angs1jk):.2f}(calced)\n"
                        sk += f"ij {ii},{jj}, {np.abs(np.mean(angs1j)-104.5)},<,angdiff_cutoff,and std {np.std(angs1j)}<{angstd_cutoff} \n"
                        sk += f"ik {ii},{kk}, {np.abs(np.mean(angs1k)-104.5)},<,angdiff_cutoff,and std {np.std(angs1k)}<{angstd_cutoff} \n"
                        sk += f"kj {kk},{jj}, {np.abs(np.mean(angskj)-104.5)},<,angdiff_cutoff,and std {np.std(angskj)}<{angstd_cutoff} \n"
                        sk += f"jk {jj},{kk}, {np.abs(np.mean(angsjk)-104.5)},<,angdiff_cutoff,and std {np.std(angsjk)}<{angstd_cutoff} \n"
                    # check if size of clusters is about equal ii==jj+kk with angle combination
                    if (
                        np.abs(N_elems - (N_elems_jj + N_elems_kk))
                        < (1 - pct_size_buffer)
                        * np.max([N_elems, N_elems_jj + N_elems_kk])
                        and (
                            np.abs(np.mean(angs1jk) - 104.5) < angdiff_cutoff
                            and np.std(angs1jk) < angstd_cutoff
                        )
                    ) or (
                        (
                            np.abs(np.mean(angs1j) - 104.5) < angdiff_cutoff
                            and np.std(angs1j) < angstd_cutoff
                        )
                        and (
                            np.abs(np.mean(angs1k) - 104.5) < angdiff_cutoff
                            and np.std(angs1k) < angstd_cutoff
                        )
                        and (
                            np.abs(np.mean(angsjk) - 104.5) < angdiff_cutoff
                            and np.std(angsjk) < angstd_cutoff
                        )
                        and (
                            np.abs(np.mean(angskj) - 104.5) < angdiff_cutoff
                            and np.std(angskj) < angstd_cutoff
                        )
                    ):
                        weakly.append([ii, jj])
                        weakly.append([ii, kk])
                        used.append(ii)
                        used.append(jj)
                        used.append(kk)
                        break
    # check if used ones account for pct_explained percentage of orientations
    total = 0
    for i in used:
        total += len(orientations[labels == i])
    if verbose > 1:
        print("total length=", total, ">", 2 * neioc * pct_explained)
        for ws in weakly:
            print("weakly combo", ws[0], ws[1])
    if total > 2 * neioc * pct_explained:
        if verbose > 0:
            print("weakly found")
        for ws in weakly:
            weakly_conserved.append(
                __return_normalized_orientation_pair(
                    orientations, labels, ws[0], ws[1], "WCW"
                )
            )
    else:  # or if angle beteween a larger set of clusters is water angle for all of them - this means that this is circular triplet or multiplet
        used = []
        weakly = []
        # loop over OPTICS clusters
        for ii in lbls:
            this = []
            # check if in used
            if ii in used:
                continue
            # average orientation of ii
            avangi = np.mean(orientations[labels == ii], axis=0)
            # loop over clusters but not ii
            lblsni = lbls[lbls != ii]
            for jj in lblsni:
                # check if already used
                if jj in used:
                    continue
                avangj = np.mean(orientations[labels == jj], axis=0)
                # loop over clusters but not ii or jj
                lblsnij = lblsni[lblsni != jj]
                angsij = (
                    np.arccos(
                        np.dot(
                            orientations[(labels == jj)],
                            avangi.T,
                        )
                    )
                    * 360.0
                    / (2.0 * np.pi)
                )
                angsji = (
                    np.arccos(
                        np.dot(
                            orientations[(labels == ii)],
                            avangj.T,
                        )
                    )
                    * 360.0
                    / (2.0 * np.pi)
                )
                if verbose > 0 or debugH > 0:
                    sk += f"ij,{ii},{jj}, angs and std ij: {np.abs(np.mean(angsij) - 104.5)}, {np.std(angsij)}; ji: {np.abs(np.mean(angsji) - 104.5)}, {np.std(angsji)} \n"
                if (
                    np.abs(np.mean(angsij) - 104.5) < angdiff_cutoff
                    and np.std(angsij) < angstd_cutoff
                ) and (
                    np.abs(np.mean(angsji) - 104.5) < angdiff_cutoff
                    and np.std(angsji) < angstd_cutoff
                ):
                    go = True
                    for kk in this:
                        avangk = np.mean(orientations[labels == kk], axis=0)
                        angsjk = (
                            np.arccos(
                                np.dot(
                                    orientations[(labels == kk)],
                                    avangj.T,
                                )
                            )
                            * 360.0
                            / (2.0 * np.pi)
                        )
                        angskj = (
                            np.arccos(
                                np.dot(
                                    orientations[(labels == jj)],
                                    avangk.T,
                                )
                            )
                            * 360.0
                            / (2.0 * np.pi)
                        )
                        if verbose > 0 or debugH > 0:
                            sk += f"jk,{jj},{kk}, angs and std jk: {np.abs(np.mean(angsjk) - 104.5)}, {np.std(angsjk)}; kj: {np.abs(np.mean(angskj) - 104.5)}, {np.std(angskj)} \n"
                        if (
                            np.abs(np.mean(angskj) - 104.5) > angdiff_cutoff
                            and np.std(angskj) > angstd_cutoff
                        ) and (
                            np.abs(np.mean(angsjk) - 104.5) > angdiff_cutoff
                            and np.std(angsjk) > angstd_cutoff
                        ):
                            go = False
                    if go:
                        weakly.append([ii, jj])
                        used.append(ii)
                        used.append(jj)
        total = 0
        for i in used:
            total += len(orientations[labels == i])
        if verbose > 2:
            print("total length=", total, ">", 2 * neioc * pct_explained)
            for ws in weakly:
                print("weakly combo", ws[0], ws[1])
        if total > 2 * neioc * pct_explained:
            if verbose > 0:
                print("weakly found")
            for ws in weakly:
                weakly_conserved.append(
                    __return_normalized_orientation_pair(
                        orientations, labels, ws[0], ws[1], "WCW"
                    )
                )

    # Debug
    if verbose > 0 or debugH > 0:
        ss = (
            f"weakly Conserved - OPTICS analysis;xi={xi}\n"
            f"Number of hydrogen clusters : {len(np.unique(labels))};\n number of elements : {counts}; range needed for best cluster:(depends on numbpct) {neioc*(2-pct_size_buffer):.2f},{neioc*lower_bound_pct_buffer:.2f}\n"
        )
    # Debug Printing
    if verbose == 2 or (verbose == 1 and len(weakly_conserved) > 0):
        print(ss + sk)
    # Debug plots
    __hydrogen_orient_plots(
        labels,
        orientations,
        cc,
        ss,
        f"Reachability of optics plot\n minsamples={int(neioc*min_samp_data_size_pct)}; xi={xi}",
        len(weakly_conserved) > 0,
        debugH,
        plotreach,
    )
    return weakly_conserved


def __hydrogen_orient_plots(
    labels, orientations, cc, ss, rtit, conserved, debugH, plotreach: bool
) -> None:
    """Collection of plots for hydrogen orientation.

    For debuging purposes. Not ment for direct usage.
    """
    if debugH == 2 or (debugH == 1 and conserved):
        if plotreach:
            fig = __plot3Dorients(111, labels, orientations, ss)
        else:
            fig = __plot3Dorients(121, labels, orientations, ss)
        if plotreach:
            __plotreachability(122, orientations, cc, fig=fig, tit=rtit)


def __plot3Dorients(subplot, labels, orientations, tip) -> Figure:
    """Function for plotting 3D orientations.

    For debuging only.

    """
    try:
        import matplotlib.pyplot as plt
    except:
        raise Exception("install matplotlib")

    fig: Figure = plt.figure()
    if type(labels) == int:
        return fig
    ax: Axes = fig.add_subplot(subplot, projection="3d")
    ax.set_title(tip)
    for j in np.unique(labels):
        jaba = orientations[labels == j]
        ax.scatter(
            jaba[:, 0],
            jaba[:, 1],
            jaba[:, 2],
            label=f"{j} ({len(labels[labels==j])})",
        )
        if j > -1:
            ax.quiver(
                0,
                0,
                0,
                np.mean(jaba[:, 0]),
                np.mean(jaba[:, 1]),
                np.mean(jaba[:, 2]),
                color="gray",
                arrow_length_ratio=0.0,
                linewidths=5,
            )
    ax.scatter(0, 0, 0, c="crimson", s=1000)
    ax.legend()
    ax.grid(False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.autoscale(tight=True)
    ax.dist = 6
    return fig


def __plotreachability(
    subplot, orientations, cc, fig: Figure | None = None, tit=None
) -> Figure:
    """Function for plotting reachability.

    For debuging purposes only.

    """
    try:
        import matplotlib.pyplot as plt
    except:
        raise Exception("install matplotlib")

    if fig is None:
        fig: Figure = plt.figure()
    if type(cc) != OPTICS:
        return fig
    lblls = cc.labels_[cc.ordering_]
    labels = cc.labels_
    reachability = cc.reachability_[cc.ordering_]
    ax2 = fig.add_subplot(subplot)
    fig.gca().set_prop_cycle(None)
    space = np.arange(len(orientations))
    ax2.plot(space, reachability)
    if tit is not None:
        ax2.set_title(tit)
    for clst in np.unique(lblls):
        if clst == -1:
            ax2.plot(
                space[lblls == clst],
                reachability[lblls == clst],
                label=f"{clst} ({len(space[lblls==clst])}), avg reach={np.mean(np.ma.masked_invalid(cc.reachability_[labels==clst]))}",
                color="blue",
            )
        else:
            ax2.plot(
                space[lblls == clst],
                reachability[lblls == clst],
                label=f"{clst} ({len(space[lblls==clst])}), avg reach={np.mean(np.ma.masked_invalid(cc.reachability_[labels==clst]))}",
            )
    ax2.legend()
    return fig


def __return_normalized_orientation_pair(
    orientations: np.ndarray,
    labels: np.ndarray,
    i: int,
    j: int,
    typel: str,
) -> list[np.ndarray | str]:
    """Helper function for normalizing orientations

    Not ment for general usage.

    """
    v1 = np.mean(orientations[labels == i], axis=0)
    v2 = np.mean(orientations[labels == j], axis=0)
    return [
        v1 / np.linalg.norm(v1) * np.linalg.norm(orientations[0]),
        v2 / np.linalg.norm(v2) * np.linalg.norm(orientations[0]),
        typel,
    ]
