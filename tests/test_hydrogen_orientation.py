"""
Unit and regression test for the ConservedWaterSearch package.
"""

# Import package, test suite, and other packages as needed
import ConservedWaterSearch.hydrogen_orientation
import pytest
import numpy as np

make_ho_plots = 0


def test_orientation_normalization():
    orientations = np.asarray([[1, 0, 0], [-0.25038 * 2, 0.96814764 * 2, 0]])
    orientations2 = np.asarray([[1, 0, 0], [-0.25038, 0.96814764, 0]])
    w1 = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations, normalize_orientations=True
    )
    w2 = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations2
    )
    print(w1, w2)
    assert w1 == w2


def test_orientation_shape():
    orientations = np.asarray([[[1]]])
    with pytest.raises(ValueError):
        ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
            orientations
        )


def test_orientation_dimensions():
    orientations = np.asarray([[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
            orientations
        )


def test_orientation_valid_input():
    orientations = np.asarray([[1, 0, 0], [0, 1, 0]])
    ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations
    )


def test_orientation_size():
    orientations = np.asarray([[1, 0, 0]])
    with pytest.raises(ValueError):
        ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
            orientations
        )


def test_orientation_array_odd():
    orientations = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
            orientations
        )


def test_conserved_FCW():
    orientations = np.loadtxt("tests/data/conserved_sample_FCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "FCW"


def test_conserved_FCW2():
    orientations = np.loadtxt("tests/data/conserved_sample_FCW2.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "FCW"


def test_dispersed_HCW():
    orientations = np.loadtxt("tests/data/dispersed_sample_HCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "HCW"


def test_dispersed_HCW2():
    orientations = np.loadtxt("tests/data/dispersed_sample_HCW2.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "HCW"


def test_circular_HCW():
    orientations = np.loadtxt("tests/data/circ_sample_HCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "HCW"


def test_circular_HCW2():
    orientations = np.loadtxt("tests/data/circ_sample_HCW2.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "HCW"


def test_circular_HCW3():
    orientations = np.loadtxt("tests/data/circ_sample_HCW3.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "HCW"


def test_WCW():
    orientations = np.loadtxt("tests/data/sample_WCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "WCW"


def test_circular_only():
    orientations = np.loadtxt("tests/data/sample_circular_only.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "WCW"


def test_dispersed_WCW():
    orientations = np.loadtxt("tests/data/dispersed_sample_WCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) > 0
    assert res[0][2] == "WCW"


def test_2x2_WCW():
    orientations = np.loadtxt("tests/data/2_by_2_WCW.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) == 2
    assert res[0][2] == "WCW"


def test_not_conserved():
    orientations = np.loadtxt("tests/data/dispersed_sample_not_conserved.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) == 0


def test_not_conserved2():
    orientations = np.loadtxt("tests/data/not_conserved.dat")
    res = ConservedWaterSearch.hydrogen_orientation.hydrogen_orientation_analysis(
        orientations,
        debugH=make_ho_plots,
        verbose=2,
    )
    if make_ho_plots > 0:
        import matplotlib.pyplot as plt

        plt.show()
    assert len(res) == 0
