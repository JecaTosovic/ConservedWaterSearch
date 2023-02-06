============
Installation
============

Installation via conda 
======================

ConservedWaterSearch is available through `conda-forge <https://conda-forge.org/>`_ using :code:`conda` (or :code:`mamba`) and this is the recommended way to install it:

.. code:: bash

   conda install -c conda-forge ConservedWaterSearch

Installation via PyPi
=====================

ConservedWaterSearch (CWS) is also available for installation from Python package index(PyPi) using :code:`pip`.

Prerequisits
------------

`Pymol <https://pymol.org/2/>`_ is required for visualisation only, if so desired. If not, it's installation can be skipped. Pymol is not available via PyPi (:code:`pip`), but can be installed from conda-forge. If pymol is already installed in your current ``python`` environment it can be used with CWS. If not, the open-source version can be installed from `conda-forge <https://conda-forge.org/>`_ via :code:`conda` (or :code:`mamba`):

.. code:: bash

   conda install -c conda-forge pymol-open-source

On the other hand `hdbscan <https://hdbscan.readthedocs.io/en/latest/index.html>`_ is a hard dependency and must be installed to use CWS. It can be installed using conda from conda-forge or from PyPi (:code:`pip`). If installing from PyPi a C++ compiler is required. It can be installed using the platform's package manager (:code:`apt` for Ubuntu, :code:`brew` for macOS, :code:`winget` for Windows) or conda:

.. code:: bash

   conda install -c conda-forge cxx-compiler

Installing the main package
---------------------------

ConservedWaterSearch can be installed via :code:`pip` from `PyPi <https://pypi.org/project/ConservedWaterSearch>`_:

.. code:: bash

   pip install ConservedWaterSearch


Known Issues
============

:code:`AttributeError: 'super' object has no attribute '_ipython_display_'`
Some versions of Jupyter notebook are incpompatible with ipython (`see here <https://stackoverflow.com/questions/74279848/nglview-installed-but-will-not-import-inside-juypter-notebook-via-anaconda-navig>`_). To resolve install version of :code:`ipywidgets<8` using :code:`conda`: 

.. code:: bash

   conda install "ipywidgets <8" -c conda-forge

or :code:`pip`:

.. code:: bash

   pip install ipywidgets==7.6.0


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
