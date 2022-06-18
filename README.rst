ConservedWaterSearch
==============================
.. image:: https://readthedocs.org/projects/conservedwatersearch/badge/?version=latest
    :target: https://conservedwatersearch.readthedocs.io/en/latest/?badge=latest
.. image:: https://badge.fury.io/py/conservedwatersearch.svg
    :target: https://badge.fury.io/py/conservedwatersearch


The conservedwatersearch Python library uses density based clustering approach to detect conserved waters from simulation trajectories.
Conserved water molecules can be further classified into 3 distinct conserved water types based on their hydrogen orientation: Fully Conserved Waters (FCW), Half Conserved Waters (HCW) and Weakly conserved waters (WCW) - see the figure below for examples.
We support many different density based clustering approaches using standard OPTICS and HDBSCAN procedures as well as multi stage re-clustering approach using either of the two algorithms for very precise (and slow) determination of conserved water molecules.

.. image:: figs/WaterTypes.png
  :width: 600


Important links
=================
	- `Documentation <https://conservedwatersearch.readthedocs.io/en/latest/>`_: hosted on Read The Docs
	- `GitHub repository <https://github.com/JecaTosovic/ConservedWaterSearch>`_: source code/contribute code
	- `Issue tracker <https://github.com/JecaTosovic/ConservedWaterSearch/issues>`_: Report issues/ request features

Related Tools
=================
	- `WaterNetworkAnalysis <https://github.com/JecaTosovic/WaterNetworkAnalysis>`_: prepare trajectories  and analyse results for/from conservedwatersearch

Citation
===============
Coming soon.

Installation
===============
The easiest ways to install **ConservedWaterSearch** is using pip:

.. code:: bash

   pip install ConservedWaterSearch

Conda builds will be available soon.


Example
===============
The easiest way to use CWS is by calling WaterNetworkAnalysis (WNA) package. However, sometimes users might want to explicitly classify conserved water molecules. A simple python code can be used to classify waters into categories given an array of 3D oxygen coordinates and their related relative hydrogen orientations:

.. code:: python

   import ConservedWaterSearch.hydrogen_orientation as HO
   # load some example
   orientations = np.loadtxt("tests/data/conserved_sample_FCW.dat")
   # Run classification
   res = HO.hydrogen_orientation_analysis(
        orientations,
   )
   # print the water type
   print(res[0][2])


For more complex usecases, please refer to the `WaterNetworkAnalysis  <https://github.com/JecaTosovic/WaterNetworkAnalysis>`_.



