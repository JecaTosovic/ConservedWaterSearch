Theory, Background, and Methods
===============================

The developed method focuses on the identification and classification of conserved water molecules and their networks derived from molecular dynamics (MD) simulations. A unique feature of this method is its ability to not only determine the positions of oxygen atoms but also the orientations of hydrogen atoms. This dual focus enables a nuanced classification into types like FCW, HCW, and WCW, which is a significant advantage over existing methods. The classification provides vital information for assessing the water network's stability. The method operates solely on data from MD simulations, without requiring any crystal water data.


Understanding Conserved Water
-----------------------------

Conserved water refers to water molecules that maintain specific, stable positions and orientations within biological systems, such as the active sites of proteins. These waters play pivotal roles in biochemical processes like protein-ligand binding and enzymatic reactions. Their presence and orientations significantly influence both the thermodynamics and kinetics of these processes.

Types of Conserved Waters in Our Approach
-----------------------------------------

Conserved waters are categorized into three main types based on hydrogen atom orientations:

- **Fully Conserved Water (FCW)**: Both hydrogen atoms exhibit a unique preferred orientation.
  
- **Half Conserved Water (HCW)**: One hydrogen atom has a unique preferred orientation, while the other hydrogen atom can have multiple orientations. HCWs are further divided into two subtypes:
  
  - **HCW-I**: One hydrogen is strongly oriented toward one acceptor, while the other switches between two different acceptors.
  - **HCW-II**: One hydrogen maintains a strong orientation toward a single acceptor, while the other moves in a circle, maintaining an optimal water angle.
  
- **Weakly Conserved Water (WCW)**: Hydrogen atoms can have several sets of preferred orientations without a single dominant one.


.. image:: https://raw.githubusercontent.com/JecaTosovic/ConservedWaterSearch/main/docs/source/figs/WaterTypes.png
  :width: 700

Applications of the Methodology
-------------------------------

The methodology for identifying conserved waters is particularly useful in protein-ligand systems to understand water networks and their effects on binding thermodynamics. The method is general enough to be applicable to water networks at any type of surface.


