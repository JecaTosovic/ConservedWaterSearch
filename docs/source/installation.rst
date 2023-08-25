============
Installation
============

The default installations from PyPI and conda-forge don't include any of the optional dependencies required for visualization (see below). This is done to avoid potential conflicts with user defined PyMOL installations. 

Installation via conda 
======================

ConservedWaterSearch is available through `conda-forge <https://conda-forge.org/>`_ using :code:`conda` (or :code:`mamba`) and this is the recommended way to install it:

.. code:: bash

   conda install -c conda-forge ConservedWaterSearch


Installation via PyPI
=====================

ConservedWaterSearch can be installed via :code:`pip` from `PyPI <https://pypi.org/project/ConservedWaterSearch>`_:

.. code:: bash

   pip install ConservedWaterSearch

Optional visualization dependencies
===================================

`PyMOL <https://PyMOL.org/2/>`_ is required for visualization only, if so desired. If not, it's installation can be skipped. PyMOL is not available via PyPI (:code:`pip`), but can be installed from conda-forge. If PyMOL is already installed in your current ``python`` environment it can be used with CWS. If not, the free (open-source) version can be installed from `conda-forge <https://conda-forge.org/>`_ via :code:`conda` (or :code:`mamba`):

.. code:: bash

   conda install -c conda-forge pymol-open-source

and paid (licensed version) from schrodinger channel (see `here <https://PyMOL.org/conda/>`_ for more details) via :code:`conda` (or :code:`mamba`):

.. code:: bash

   conda install -c conda-forge -c schrodinger pymol-bundle

Optional dependencies for visualization can be installed through PyPI (:code:`pip`) except for PyMOL. To install CWS through :code:`pip` with nglview use:

.. code:: bash

   pip install ConservedWaterSearch[nglview]

To install with matplotlib for clustering plots use:

.. code:: bash

   pip install ConservedWaterSearch[debug]

To install with both nglview and matplotlib use:

.. code:: bash

   pip install ConservedWaterSearch[all]

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

Or via PyPI (`see here <https://pypi.org/project/pytest>`_) (using :code:`pip`):

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
