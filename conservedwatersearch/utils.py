import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import OPTICS


def hydrogen_orient_plots(
    labels, orientations, cc, ss, rtit, conserved, debugH, plotreach
):
    """Collection of plots for hydrogen orientation"""
    if debugH == 2 or (debugH == 1 and conserved):
        if plotreach:
            fig = plot3Dorients(111, labels, orientations, ss)
        else:
            fig = plot3Dorients(121, labels, orientations, ss)
        if plotreach:
            plotreachability(122, orientations, cc, fig=fig, tit=rtit)


def plot3Dorients(subplot, labels, orientations, tip):
    fig = plt.figure()
    if type(labels) == int:
        return fig
    ax = fig.add_subplot(subplot, projection="3d")
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


def plotreachability(subplot, orientations, cc, fig=None, tit=None):
    if fig is None:
        fig = plt.figure()
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
