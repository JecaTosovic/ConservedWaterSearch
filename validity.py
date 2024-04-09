# %%
import time
import numpy as np
from numpy import isclose
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster._hdbscan._linkage import (
    mst_from_data_matrix,
    mst_from_mutual_reachability,
)
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.cluster._hdbscan._tree import _condense_tree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph
from functools import reduce, partial


def DBCV(X, labels, dist_function=euclidean):
    """
    Density Based clustering validation

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: cluster_validity (float)
        score in range[-1, 1] indicating validity of clustering assignments
    """
    graph = _mutual_reach_dist_graph(X, labels, dist_function)
    mst = _mutual_reach_dist_MST(graph)
    cluster_validity = _clustering_validity_index(mst, labels)
    return cluster_validity


def _mutual_reach_dist_graph(X, labels, dist_function):
    """
    Computes the mutual reach distance complete graph.
    Graph of all pair-wise mutual reachability distances between points

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: graph (np.ndarray)
        array of dimensions (n_samples, n_samples)
        Graph of all pair-wise mutual reachability distances between points.

    """
    if isinstance(dist_function, int):
        n_samples = np.shape(X)[0]
        n_features = dist_function
        dists = X
        graph = np.empty_like(X)
    else:
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]
        dists = cdist(X, X, dist_function)
        # If we're calculating the distances ourselves, might as well reuse the array for later
        graph = dists

    core_dists = np.empty(n_samples)
    for label in np.unique(labels):
        mask = labels == label
        distance_vectors = dists[mask, :][:, mask]
        n_neighbors = distance_vectors.shape[0]
        z = distance_vectors == 0
        distance_vectors[z] = np.nan
        numerator = np.nansum((1 / distance_vectors) ** n_features, axis=1)
        cluster_core_dists = (numerator / (n_neighbors - 1)) ** (-1 / n_features)
        core_dists[mask] = cluster_core_dists

    distances = np.broadcast_arrays(
        dists, core_dists[:, np.newaxis], core_dists[np.newaxis, :]
    )
    reduce(partial(np.maximum, out=graph), distances)
    return graph


def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the mutual reach distance complete graph

    Args:
        dist_tree (np.ndarray): array of dimensions (n_samples, n_samples)
            Graph of all pair-wise mutual reachability distances
            between points.

    Returns: minimum_spanning_tree (np.ndarray)
        array of dimensions (n_samples, n_samples)
        minimum spanning tree of all pair-wise mutual reachability
            distances between points.
    """
    mst = minimum_spanning_tree(dist_tree).toarray()
    return mst + np.transpose(mst)


def _cluster_density_sparseness(MST, labels, cluster):
    """
    Computes the cluster density sparseness, the minimum density
        within a cluster

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_density_sparseness (float)
        value corresponding to the minimum density within a cluster
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[indices][:, indices]
    cluster_density_sparseness = np.max(cluster_MST)
    return cluster_density_sparseness


def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Computes the density separation between two clusters, the maximum
        density between clusters.

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster_i (int): cluster i of interest
        cluster_j (int): cluster j of interest

    Returns: density_separation (float):
        value corresponding to the maximum density between clusters
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
    relevant_paths = shortest_paths[:, indices_j]
    density_separation = np.min(relevant_paths)
    return density_separation


def _cluster_validity_index(MST, labels, cluster):
    """
    Computes the validity of a cluster (validity of assignmnets)

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_validity (float)
        value corresponding to the validity of cluster assignments
    """
    min_density_separation = np.inf
    for cluster_j in np.unique(labels):
        if cluster_j != cluster:
            cluster_density_separation = _cluster_density_separation(
                MST, labels, cluster, cluster_j
            )
            if cluster_density_separation < min_density_separation:
                min_density_separation = cluster_density_separation
    cluster_density_sparseness = _cluster_density_sparseness(MST, labels, cluster)
    numerator = min_density_separation - cluster_density_sparseness
    denominator = np.max([min_density_separation, cluster_density_sparseness])
    cluster_validity = numerator / denominator
    return cluster_validity


def _clustering_validity_index(MST, labels):
    """
    Computes the validity of all clustering assignments for a
    clustering algorithm

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X

    Returns: validity_index (float):
        score in range[-1, 1] indicating validity of clustering assignments
    """
    n_samples = len(labels)
    validity_index = 0
    for label in np.unique(labels):
        fraction = np.sum(labels == label) / float(n_samples)
        cluster_validity = _cluster_validity_index(MST, labels, label)
        validity_index += fraction * cluster_validity
    return validity_index


################# HDBSCAN's validity
def all_points_core_distance(distance_matrix, d=2.0):
    """
    Compute the all-points-core-distance for all the points of a cluster.

    Parameters
    ----------
    distance_matrix : array (cluster_size, cluster_size)
        The pairwise distance matrix between points in the cluster.

    d : integer
        The dimension of the data set, which is used in the computation
        of the all-point-core-distance as per the paper.

    Returns
    -------
    core_distances : array (cluster_size,)
        The all-points-core-distance of each point in the cluster

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    distance_matrix[distance_matrix != 0] = (
        1.0 / distance_matrix[distance_matrix != 0]
    ) ** d
    result = distance_matrix.sum(axis=1)
    result /= distance_matrix.shape[0] - 1

    if result.sum() == 0:
        result = np.zeros(len(distance_matrix))
    else:
        result **= -1.0 / d

    return result


def max_ratio(stacked_distances):
    # Extract the distances and core distances
    distances = stacked_distances[:, :, 0]
    coredists = stacked_distances[:, :, 1]

    # Replace zeros in distances with ones to avoid division by zero
    distances[distances == 0] = 1

    # Compute the ratios
    ratios = coredists / distances

    # Find the maximum ratio greater than zero
    max_ratio = ratios[ratios > 0].max()

    return max_ratio


def distances_between_points(
    X,
    labels,
    cluster_id,
    metric="euclidean",
    d=None,
    no_coredist=False,
    print_max_raw_to_coredist_ratio=False,
    **kwd_args
):
    """
    Compute pairwise distances for all the points of a cluster.

    If metric is 'precomputed' then assume X is a distance matrix for the full
    dataset. Note that in this case you must pass in 'd' the dimension of the
    dataset.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    cluster_id : integer
        The cluster label for which to compute the distances

    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    d : integer (or None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------

    distances : array (n_samples, n_samples)
        The distances between all points in `X` with `label` equal to `cluster_id`.

    core_distances : array (n_samples,)
        The all-points-core_distance of all points in `X` with `label` equal
        to `cluster_id`.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == "precomputed":
        if d is None:
            raise ValueError("If metric is precomputed a " "d value must be provided!")
        distance_matrix = X[labels == cluster_id, :][:, labels == cluster_id]
    else:
        subset_X = X[labels == cluster_id, :]
        distance_matrix = pairwise_distances(subset_X, metric=metric, **kwd_args)
        d = X.shape[1]

    if no_coredist:
        return distance_matrix, None

    else:
        core_distances = all_points_core_distance(distance_matrix.copy(), d=d)
        core_dist_matrix = np.tile(core_distances, (core_distances.shape[0], 1))
        stacked_distances = np.dstack(
            [distance_matrix, core_dist_matrix, core_dist_matrix.T]
        )

        if print_max_raw_to_coredist_ratio:
            print(
                "Max raw distance to coredistance ratio: "
                + str(max_ratio(stacked_distances))
            )

        return stacked_distances.max(axis=-1), core_distances


def convert_mst_output(mst_edges):
    result = np.zeros((mst_edges.shape[0], 3))
    result[:, 0] = mst_edges["current_node"]
    result[:, 1] = mst_edges["next_node"]
    result[:, 2] = mst_edges["distance"]
    return result


def internal_minimum_spanning_tree(mr_distances):
    """
    Compute the 'internal' minimum spanning tree given a matrix of mutual
    reachability distances. Given a minimum spanning tree the 'internal'
    graph is the subgraph induced by vertices of degree greater than one.

    Parameters
    ----------
    mr_distances : array (cluster_size, cluster_size)
        The pairwise mutual reachability distances, inferred to be the edge
        weights of a complete graph. Since MSTs are computed per cluster
        this is the all-points-mutual-reacability for points within a single
        cluster.

    Returns
    -------
    internal_nodes : array
        An array listing the indices of the internal nodes of the MST

    internal_edges : array (?, 3)
        An array of internal edges in weighted edge list format; that is
        an edge is an array of length three listing the two vertices
        forming the edge and weight of the edge.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    # Compute the MST using mst_from_mutual_reachability
    mst_edges = mst_from_mutual_reachability(mr_distances)

    # Convert to the desired format
    single_linkage_data = convert_mst_output(mst_edges)

    min_span_tree = single_linkage_data.copy()
    for index, row in enumerate(min_span_tree[1:], 1):
        candidates = np.where(isclose(mr_distances[int(row[1])], row[2]))[0]
        candidates = np.intersect1d(
            candidates, single_linkage_data[:index, :2].astype(int)
        )
        candidates = candidates[candidates != row[1]]
        assert len(candidates) > 0
        row[0] = candidates[0]

    vertices = np.arange(mr_distances.shape[0])[
        np.bincount(min_span_tree.T[:2].flatten().astype(np.intp)) > 1
    ]
    if not len(vertices):
        vertices = [0]
    # A little "fancy" we select from the flattened array reshape back
    # (Fortran format to get indexing right) and take the product to do an and
    # then convert back to boolean type.
    edge_selection = np.prod(
        np.in1d(min_span_tree.T[:2], vertices).reshape(
            (min_span_tree.shape[0], 2), order="F"
        ),
        axis=1,
    ).astype(bool)

    # Density sparseness is not well defined if there are no
    # internal edges (as per the referenced paper). However
    # MATLAB code from the original authors simply selects the
    # largest of *all* the edges in the case that there are
    # no internal edges, so we do the same here
    if np.any(edge_selection):
        # If there are any internal edges, then subselect them out
        edges = min_span_tree[edge_selection]
    else:
        # If there are no internal edges then we want to take the
        # max over all the edges that exist in the MST, so we simply
        # do nothing and return all the edges in the MST.
        edges = min_span_tree.copy()

    return vertices, edges


def density_separation(
    X,
    labels,
    cluster_id1,
    cluster_id2,
    internal_nodes1,
    internal_nodes2,
    core_distances1,
    core_distances2,
    metric="euclidean",
    no_coredist=False,
    **kwd_args
):
    """
    Compute the density separation between two clusters. This is the minimum
    distance between pairs of points, one from internal nodes of MSTs of each cluster.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    cluster_id1 : integer
        The first cluster label to compute separation between.

    cluster_id2 : integer
        The second cluster label to compute separation between.

    internal_nodes1 : array
        The vertices of the MST for `cluster_id1` that were internal vertices.

    internal_nodes2 : array
        The vertices of the MST for `cluster_id2` that were internal vertices.

    core_distances1 : array (size of cluster_id1,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id1.

    core_distances2 : array (size of cluster_id2,)
        The all-points-core_distances of all points in the cluster
        specified by cluster_id2.

    metric : string
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    The 'density separation' between the clusters specified by
    `cluster_id1` and `cluster_id2`.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    if metric == "precomputed":
        sub_select = X[labels == cluster_id1, :][:, labels == cluster_id2]
        distance_matrix = sub_select[internal_nodes1, :][:, internal_nodes2]
    else:
        cluster1 = X[labels == cluster_id1][internal_nodes1]
        cluster2 = X[labels == cluster_id2][internal_nodes2]
        distance_matrix = cdist(cluster1, cluster2, metric, **kwd_args)

    if no_coredist:
        return distance_matrix.min()

    else:
        core_dist_matrix1 = np.tile(
            core_distances1[internal_nodes1], (distance_matrix.shape[1], 1)
        ).T
        core_dist_matrix2 = np.tile(
            core_distances2[internal_nodes2], (distance_matrix.shape[0], 1)
        )

        mr_dist_matrix = np.dstack(
            [distance_matrix, core_dist_matrix1, core_dist_matrix2]
        ).max(axis=-1)

        return mr_dist_matrix.min()


def validity_index(
    X,
    labels,
    metric="euclidean",
    d=None,
    per_cluster_scores=False,
    mst_raw_dist=False,
    verbose=False,
    **kwd_args
):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples, n_samples)
        The input data of the clustering. This can be the data, or, if
        metric is set to `precomputed` the pairwise distance matrix used
        for the clustering.

    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    metric : optional, string (default 'euclidean')
        The metric used to compute distances for the clustering (and
        to be re-used in computing distances for mr distance). If
        set to `precomputed` then X is assumed to be the precomputed
        distance matrix between samples.

    d : optional, integer (or None) (default None)
        The number of features (dimension) of the dataset. This need only
        be set in the case of metric being set to `precomputed`, where
        the ambient dimension of the data is unknown to the function.

    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.

    mst_raw_dist : optional, boolean (default False)
        If True, the MST's are constructed solely via 'raw' distances (depending on the given metric, e.g. euclidean distances)
        instead of using mutual reachability distances. Thus setting this parameter to True avoids using 'all-points-core-distances' at all.
        This is advantageous specifically in the case of elongated clusters that lie in close proximity to each other <citation needed>.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.

    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    core_distances = {}
    density_sparseness = {}
    mst_nodes = {}
    mst_edges = {}

    max_cluster_id = labels.max() + 1
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id), dtype=np.float64)
    cluster_validity_indices = np.empty(max_cluster_id, dtype=np.float64)

    for cluster_id in range(max_cluster_id):
        if np.sum(labels == cluster_id) == 0:
            continue

        distances_for_mst, core_distances[cluster_id] = distances_between_points(
            X,
            labels,
            cluster_id,
            metric,
            d,
            no_coredist=mst_raw_dist,
            print_max_raw_to_coredist_ratio=verbose,
            **kwd_args
        )

        mst_nodes[cluster_id], mst_edges[cluster_id] = internal_minimum_spanning_tree(
            distances_for_mst
        )
        density_sparseness[cluster_id] = mst_edges[cluster_id].T[2].max()

    for i in range(max_cluster_id):
        if np.sum(labels == i) == 0:
            continue

        internal_nodes_i = mst_nodes[i]
        for j in range(i + 1, max_cluster_id):
            if np.sum(labels == j) == 0:
                continue

            internal_nodes_j = mst_nodes[j]
            density_sep[i, j] = density_separation(
                X,
                labels,
                i,
                j,
                internal_nodes_i,
                internal_nodes_j,
                core_distances[i],
                core_distances[j],
                metric=metric,
                no_coredist=mst_raw_dist,
                **kwd_args
            )
            density_sep[j, i] = density_sep[i, j]

    n_samples = float(X.shape[0])
    result = 0

    for i in range(max_cluster_id):
        if np.sum(labels == i) == 0:
            continue

        min_density_sep = density_sep[i].min()
        cluster_validity_indices[i] = (min_density_sep - density_sparseness[i]) / max(
            min_density_sep, density_sparseness[i]
        )

        if verbose:
            print("Minimum density separation: " + str(min_density_sep))
            print("Density sparseness: " + str(density_sparseness[i]))

        cluster_size = np.sum(labels == i)
        result += (cluster_size / n_samples) * cluster_validity_indices[i]

    if per_cluster_scores:
        return result, cluster_validity_indices
    else:
        return result


def relative_validity_index_from_SLT(
    labels,
    single_linkage_tree,
    per_cluster_scores=False,
):
    """
    Compute the density based cluster validity index for the
    clustering specified by `labels` and for each cluster in `labels`.

    Parameters
    ----------
    labels : array (n_samples)
        The label array output by the clustering, providing an integral
        cluster label to each data point, with -1 for noise points.

    single_linkage_tree : array (n_samples - 1, 4)
        The single linkage tree output by the clustering, providing the
        hierarchical clustering of the data.

    per_cluster_scores : optional, boolean (default False)
        Whether to return the validity index for individual clusters.
        Defaults to False with the function returning a single float
        value for the whole clustering.

    **kwd_args :
        Extra arguments to pass to the distance computation for other
        metrics, such as minkowski, Mahanalobis etc.

    Returns
    -------
    validity_index : float
        The density based cluster validity index for the clustering. This
        is a numeric value between -1 and 1, with higher values indicating
        a 'better' clustering.

    per_cluster_validity_index : array (n_clusters,)
        The cluster validity index of each individual cluster as an array.
        The overall validity index is the weighted average of these values.
        Only returned if per_cluster_scores is set to True.

    References
    ----------
    Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
    2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
    """
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_size = sizes[1:]
    total = noise_size + np.sum(cluster_size)
    num_clusters = len(cluster_size)
    DSC = np.zeros(num_clusters)
    min_outlier_sep = np.inf  # only required if num_clusters = 1
    correction_const = 2  # only required if num_clusters = 1

    # Unltimately, for each Ci, we only require the
    # minimum of DSPC(Ci, Cj) over all Cj != Ci.
    # So let's call this value DSPC_wrt(Ci), i.e.
    # density separation 'with respect to' Ci.
    DSPC_wrt = np.ones(num_clusters) * np.inf
    max_distance = 0

#this didnt help
    mst = _condense_tree(single_linkage_tree.copy(), 5)
    print(mst[:5])
  #  for p1, p2, length, p3 in single_linkage_tree:
  #      print(p1,p2,length,p3)
        #if np.any(np.asarray([p1,p2,p3])>len(labels)):
        #    print(p1,p2,p3)
    
    #THE ISSUE HERE IS THAT HDBSCANS SLT from scikit learn is not the
    #same as the one from hdbscan contrib. The hdbscan contrib has node
    #in node out, while the scikit learn one has some other node stuff
    #relate dto condensed tree and contains praent/child pairs and
    #cluster size

    for p1, p2, length, _ in mst:
        max_distance = max(max_distance, length)
        label1 = labels[p1]
        label2 = labels[p2]

        if label1 == -1 and label2 == -1:
            continue
        elif label1 == -1 or label2 == -1:
            # If exactly one of the points is noise
            min_outlier_sep = min(min_outlier_sep, length)
            continue

        if label1 == label2:
            # Set the density sparseness of the cluster
            # to the sparsest value seen so far.
            DSC[label1] = max(length, DSC[label1])
        else:
            # Check whether density separations with
            # respect to each of these clusters can
            # be reduced.
            DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
            DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

    # In case min_outlier_sep is still np.inf, we assign a new value to it.
    # This only makes sense if num_clusters = 1 since it has turned out
    # that the MR-MST has no edges between a noise point and a core point.
    min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

    # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
    # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
    # MR-MST might contain an edge with one point in Cj and ther other one
    # in Ck. Here, we replace the infinite density separation of Ci by
    # another large enough value.
    #
    # TODO: Think of a better yet efficient way to handle this.
    correction = correction_const * (
        max_distance if num_clusters > 1 else min_outlier_sep
    )
    DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

    V_index = [
        (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
        for i in range(num_clusters)
    ]
    cluster_scores = np.array(
        [cluster_size[i] * V_index[i] for i in range(num_clusters)]
    )
    score = np.sum(
        cluster_scores / total
    )
    if per_cluster_scores:
        return score, cluster_scores
    else:
        return score
        
#from sklearn.datasets import make_blobs, make_moons
#from sklearn.cluster import HDBSCAN
#X_m, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
#clustering_m_1 = HDBSCAN().fit(X_m)
#print(clustering_m_1._single_linkage_tree_[:5])
#print("Validity index moons HDBSCAN: " + str(relative_validity_index_from_SLT(clustering_m_1.labels_, clustering_m_1._single_linkage_tree_)))

#%%
def test_normal(plot=False):
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

    # Create a dataset with 3 well-separated clusters
    X_b, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    X_m, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # Apply DBSCAN clustering
    clustering_b_1 = HDBSCAN().fit(X_b)
    clustering_b_2 = DBSCAN(eps=0.75).fit(X_b)
    clustering_b_3 = OPTICS().fit(X_b)
    # Compute the validity index
    validity_index_result_b_1 = validity_index(X_b, clustering_b_1.labels_, mst_raw_dist=True)
    validity_index_result_b_2 = validity_index(X_b, clustering_b_2.labels_, mst_raw_dist=True)
    validity_index_result_b_3 = validity_index(X_b, clustering_b_3.labels_, mst_raw_dist=True)
    print("Validity index blobs HDBSCAN: " + str(validity_index_result_b_1))
    print("Validity index blobs DBSCAN eps=0.75: " + str(validity_index_result_b_2))
    print("Validity index blobs OPTICS: " + str(validity_index_result_b_3))
    DBCV_b_1 = DBCV(X_b, clustering_b_1.labels_)
    DBCV_b_2 = DBCV(X_b, clustering_b_2.labels_)
    DBCV_b_3 = DBCV(X_b, clustering_b_3.labels_)
    print(" DBCV blobs HDBSCAN: " + str(DBCV_b_1))
    print(" DBCV blobs DBSCAN eps=0.75: " + str(DBCV_b_2))
    print(" DBCV blobs OPTICS: " + str(DBCV_b_3))
    clustering_m_1 = HDBSCAN().fit(X_m)
    clustering_m_2 = DBSCAN(eps=0.75).fit(X_m)
    clustering_m_3 = OPTICS().fit(X_m)
    # Compute the validity index
    validity_index_result_m_1 = validity_index(X_m, clustering_m_1.labels_, mst_raw_dist=True)
    validity_index_result_m_2 = validity_index(X_m, clustering_m_2.labels_, mst_raw_dist=True)
    validity_index_result_m_3 = validity_index(X_m, clustering_m_3.labels_, mst_raw_dist=True)
    print("Validity index moons HDBSCAN: " + str(validity_index_result_m_1))
    print("Validity index moons DBSCAN eps=0.75: " + str(validity_index_result_m_2))
    print("Validity index moons OPTICS: " + str(validity_index_result_m_3))
    DBCV_m_1 = DBCV(X_m, clustering_m_1.labels_)
    DBCV_m_2 = DBCV(X_m, clustering_m_2.labels_)
    DBCV_m_3 = DBCV(X_m, clustering_m_3.labels_)
    print(" DBCV moons HDBSCAN: " + str(DBCV_m_1))
    print(" DBCV moons DBSCAN eps=0.75: " + str(DBCV_m_2))
    print(" DBCV moons OPTICS: " + str(DBCV_m_3))
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        plt.subplot(2, 3, 1)
        plt.scatter(X_b[:, 0], X_b[:, 1], c=clustering_b_1.labels_, cmap="viridis", s=50)
        plt.title(
            "HDBSCAN; \nvalidity index: "
            + str(validity_index_result_b_1)
            + " \nDBCV: "
            + str(DBCV_b_1)
        )

        plt.subplot(2, 3, 2)
        plt.scatter(X_b[:, 0], X_b[:, 1], c=clustering_b_2.labels_, cmap="viridis", s=50)
        plt.title(
            "DBSCAN eps=0.75;\nvalidity index: "
            + str(validity_index_result_b_2)
            + " \nDBCV: "
            + str(DBCV_b_2)
        )

        plt.subplot(2, 3, 3)
        plt.scatter(X_b[:, 0], X_b[:, 1], c=clustering_b_3.labels_, cmap="viridis", s=50)
        plt.title(
            "OPTICS;\nvalidity index: "
            + str(validity_index_result_b_3)
            + " \nDBCV: "
            + str(DBCV_b_3)
        )

        plt.subplot(2, 3, 4)
        plt.scatter(X_m[:, 0], X_m[:, 1], c=clustering_m_1.labels_, cmap="viridis", s=50)
        plt.title(
            "HDBSCAN; \nvalidity index: "
            + str(validity_index_result_m_1)
            + " \nDBCV: "
            + str(DBCV_m_1)
        )

        plt.subplot(2, 3, 5)
        plt.scatter(X_m[:, 0], X_m[:, 1], c=clustering_m_2.labels_, cmap="viridis", s=50)
        plt.title(
            "DBSCAN eps=0.75;\nvalidity index: "
            + str(validity_index_result_m_2)
            + " \nDBCV: "
            + str(DBCV_m_2)
        )

        plt.subplot(2, 3, 6)
        plt.scatter(X_m[:, 0], X_m[:, 1], c=clustering_m_3.labels_, cmap="viridis", s=50)
        plt.title(
            "OPTICS;\nvalidity index: "
            + str(validity_index_result_m_3)
            + " \nDBCV: "
            + str(DBCV_m_3)
        )
        # y space between plots should be twice as much
        plt.subplots_adjust(hspace=0.6)        
        plt.show()


# if main
if __name__ == "__main__":
    test_normal(True)

#%%
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Create a dataset with 3 well-separated clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply DBSCAN clustering
clustering = DBSCAN(eps=0.5).fit(X)
labels = clustering.labels_
%timeit validity_index(X, labels)
%timeit DBCV(X, labels)
