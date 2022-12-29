============
Installation
============

Installation via conda 
======================

ConservedWaterSearch is currently not available through conda, but will be soon.

All the dependencies can be installed from `conda-forge <https://conda-forge.org/>`_ using :code:`conda` (or :code:`mamba`):

.. code:: bash

   conda install -c conda-forge numpy matplotlib scikit-learn hdbscan pymol-open-source nglview

Installation via PyPi
=====================

ConservedWaterSearch (CWS) is available for installation from Python package index(PyPi).

Prerequisits
------------

`Pymol <https://pymol.org/2/>`_ is required for visualisation only, if so desired. However, pymol is not available via PyPi (:code:`pip`), but can be installed from conda-forge. If pymol is already installed in your current ``python`` environment it can be used with CWS. If not, the open-source version can be installed from `conda-forge <https://conda-forge.org/>`_ via :code:`conda` (or :code:`mamba`):

.. code:: bash

   conda install -c conda-forge pymol-open-source

Installing the main package
---------------------------

ConservedWaterSearch can be installed via :code:`pip` from `PyPi <https://pypi.org/project/ConservedWaterSearch>`_:

.. code:: bash

   pip install ConservedWaterSearch


Tests
=====

ConservedWaterSearch (CWS) uses :code:`pytest` for running unit tests. It can be installed via :code:`conda`:

.. code:: bash

   conda install -c conda-forge pytest

Or via PyPi (`see here <https://pypi.org/project/pytest>`_) (using :code:`pip`):

.. code:: bash

   pip install pytest

Unit tests can be run from the root directory:

.. code:: bash

   python -m pytest

Documentation
=============

To build the documentation following dependencies have to be installed, either using :code:`conda`:

.. code:: bash

   conda install -c conda-forge sphinx sphinx_rtd_theme sphinxcontrib-bibtex

or :code:`pip`:

.. code:: bash

   pip install sphinx sphinx_rtd_theme sphinxcontrib-bibtex

The documentation can be build from the :code:`docs` folder:

.. code:: bash

   cd docs/
   make html
   open build/html/index.html
