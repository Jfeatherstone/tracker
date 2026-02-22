``tracker``
============

This library includes the computational tools associated with Featherstone
et al. (2026). Beyond sharing the code used in analyzing the data associated
data, we also include a vast array of tutorials about trajectory analysis
and simulation.

These tutorials assume some basic knowledge of Python and statistical physics,
though we do our best to make them as approachable and understandable as possible.


Installation
------------

The package can be installed from source:

.. code-block:: console

    $ git clone https://github.com/jfeatherstone/tracker
    $ cd tracker
    $ pip install .

This library depends on the usual scientific computing libraries that you
probably already have installed: ``numpy``, ``scipy``, ``matplotlib``.

The package ``tqdm`` is used for creating progress bars.


Usage
-----

The API documentation describes how to directly use the library if you are
already familiar with Python.

If you're interested in recreating a similar analysis to that which is presented
in the associated publication, it is recommended to read through the tutorials,
as they explicitly describe how to do all of these things.


Tutorials
---------

All of the tutorials are written assuming you are on a linux system (since
that is what I use). In theory, other operating systems should be almost
exactly with the exception of the first tutorial on setting things up.


Indices and tables
------------------

.. toctree::
    :maxdepth: 1

    tutorials_top
    derivations_top

* :ref:`genindex`
* :ref:`search`

