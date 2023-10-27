Theory, Background, and Methods
===============================

Understanding Conserved Water Molecules
------------------------------

Conserved water refers to water molecules that maintain specific, stable positions and orientations within biological systems, such as the active sites of proteins. These waters play pivotal roles in biochemical processes like protein-ligand binding and enzymatic reactions. Their presence and orientations significantly influence both the thermodynamics and kinetics of these processes. The importance of these water networks in influencing the thermodynamic signature of ligand-protein binding has been the subject of several studies. For example, researchers studied how water networks affect the thermodynamics of binding between phosphonopeptide inhibitors and the enzyme thermolysin. They found a direct correlation between the structure of the water networks and the thermodynamic profiles of the binding interactions :cite:`Betz2016,ENGLERT2010,Biela2012,Biela2013,Krimmer2014,Krimmer2016,Cramer2017`. In another study, mutations that disrupted nearby water networks were found to lower the affinity of ligand-protein binding in the *Haemophilus influenzae virulence* protein SiaP :cite:`Darby2019`. These findings highlight that the stabilization of water molecules through water network formation leads to more favorable binding signatures, emphasizing the critical role that conserved water plays in biochemical interactions.



Overview of the Methodology
-----------------------------

The developed method focuses on the identification and classification of conserved water molecules and their networks derived from molecular dynamics (MD) simulations :cite:`conservedwatersearch2022`. A distinguishing aspect of this approach is its two-fold analytical capability: it not only pinpoints the positions of oxygen atoms but also performs a novel analysis of hydrogen orientations. Additionally, the method incorporates a residence time criterion. It ensures that the water molecules in selected clusters are present in their respective positions for the majority of the simulation time, thus qualifying them as conserved water molecules. Working in conjunction with the hydrogen orientation analysis, this residence time criterion provides a robust classification of water molecules into three distinct types: Fully Conserved Water (FCW), Half Conserved Water (HCW), and Weakly Conserved Water (WCW), based on their preferred hydrogen orientations toward the receptor. This classification is a significant advancement over existing methods and provides critical insights into the roles and contributions of individual waters in the network. As a result, it offers a more comprehensive understanding of the water network's stability. The method is particularly useful in protein-ligand systems to understand water networks and their effects on binding thermodynamics, but it is general enough to be applicable to water networks at any type of surface. The method relies exclusively on data from MD simulations, eliminating the need for any crystal water data.





Types of Conserved Waters in Our Approach
-----------------------------------------

Conserved waters are categorized into three main types based on hydrogen atom orientations:

- **Fully Conserved Water (FCW)**: Both hydrogen atoms exhibit a unique preferred orientation.
  
- **Half Conserved Water (HCW)**: One hydrogen atom has a unique preferred orientation, while the other hydrogen atom can have multiple orientations. HCWs are further divided into two subtypes:
  
  - **HCW-I**: One hydrogen is strongly oriented toward one acceptor, while the other switches between two different acceptors.
  - **HCW-II**: One hydrogen maintains a strong orientation toward a single acceptor, while the other moves in a circle, maintaining an optimal water angle.
  
- **Weakly Conserved Water (WCW)**: Hydrogen atoms can have several sets of preferred orientations without a single dominant one. WCWs can have different configurations, such as doublets (WCW-I), triplet (WCW-II), or a circular orientation (WCW-III).


.. image:: https://raw.githubusercontent.com/JecaTosovic/ConservedWaterSearch/main/docs/source/figs/WaterTypes.png
  :width: 700


FCW and HCW water molecules are more likely to represent real conserved water molecules in the system compared to WCW. This is attributed to the single preferred hydrogen orientation in both FCW and HCW, which presents an additional stringent criterion (besides the cluster size of oxygen clusters) that indirectly mandates a residence time for these molecules to be very close to 100%.

On the contrary, WCW molecules possess a considerably lower likelihood of representing conserved water molecules. This discrepancy primarily originates from the lax orientation criteria associated with WCW molecules, which, in contrast to the rigid preferred hydrogen orientations in FCW and HCW, allow for multiple preferred orientations. This leniency in hydrogen orientation lacks a stringent criterion that could indirectly enforce a high residence time, a characteristic indicative of conserved water molecules.

Additionally, the fundamental condition for WCW clustering is centered around maintaining comparable cluster sizes and ensuring an acceptable water angle between clusters, as opposed to adhering to a specific hydrogen orientation. This condition, although encouraging a variety of orientations, falls short in ascertaining a high residence time, thereby not guaranteeing the conservation of water molecules within the system.

Furthermore, the flexible orientation criteria for WCW molecules make them prone to potential mobility and variability in occupying designated positions within the clusters. Such mobility may result in different water molecules sporadically assuming the designated WCW positions, consequently undermining the reliability and conservation status of these molecules within the system. This variability, coupled with the absence of a rigid hydrogen orientation criterion, diminishes the confidence level in WCW molecules representing real conserved water molecules, rendering them less reliable in this aspect compared to FCW and HCW molecules.


.. rubric:: References:
.. bibliography:: references/references.bib