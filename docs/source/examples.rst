========
Examples
========

Classification of conserved water types
---------------------------------------

Sometimes users might want to explicitly classify conserved water molecules. A simple python code can be used to classify waters into categories given an array of 3D oxygen coordinates and their related relative hydrogen orientations:

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

Checking specific water types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Preprocessing of trajectories using WaterNetworkAnalysis
--------------------------------------------------------



Clustering of watesr via oxygencls
----------------------------------

For more information on preprocessing trajectory data, please refer to the `WaterNetworkAnalysis  <https://github.com/JecaTosovic/WaterNetworkAnalysis>`_.

The easiest way to use CWS is by calling :code:`WaterClustering` class. The starting trajectory should be aligned first, and coordinates of water oxygen and hydrogens extracted. See `WaterNetworkAnalysis  <https://github.com/JecaTosovic/WaterNetworkAnalysis>`_ for more information and convenience functions.

.. code:: python

   # imports
   from ConservedWaterSearch.water_clustering import WaterClustering
   from ConservedWaterSearch.utils import get_orientations_from_positions
   # Number of snapshots
   Nsnap = 20
   # load some example - trajectory should be aligned prior to extraction of atom coordinates
   Opos = np.loadtxt("tests/data/testdataO.dat")
   Hpos = np.loadtxt("tests/data/testdataH.dat")
   wc = WaterClustering(nsnaps=Nsnap, save_intermediate_results=False, save_results_after_done=False)
   wc.multi_stage_reclustering(*get_orientations_from_positions(Opos, Hpos))
   print(wc.water_type)
   # "aligned.pdb" should be the snapshot original trajectory was aligned to.
   wc.visualise_pymol(aligned_protein = "aligned.pdb", output_file = "waters.pse")