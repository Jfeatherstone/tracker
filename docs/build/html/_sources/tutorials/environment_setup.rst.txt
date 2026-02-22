Setup your environment
----------------------

This page covers how to setup your Python environment in order to use the
utilities provided by ``tracker`` and/or follow the rest of the tutorials.
If you are familiar with Python and already have an environment setup,
feel free to skip this tutorial.

I recommend having two separate Python environments in order to follow
the exact analysis process described in the paper: one in order to do the
neural network tracking with `SLEAP <https://sleap.ai>`_, and another for all of the subsequent
analyses. It is best to separate these two because SLEAP has rather particular
requirements about which packages it uses, and I've found it to be more
convenient to isolate it in its own environment than to try to wrangle package
versions. Further, typically one runs the neural network training and tracking
on a computer with a GPU, as it will be *much* faster. The analyses can be
run on pretty much any modern computer, so you may have these two environments
installed on different machines (eg. for me, the SLEAP environment on a
cluster, and the analyses on my usual laptop).

The package is setup to be used either with or without SLEAP installed, so
you can access the analysis methods without SLEAP in the latter environment.

Install conda
~~~~~~~~~~~~~

We will use conda to manage the different Python environments, which you can
install by downloading the appropriate file from `miniforge <https://conda-forge.org/download/>`_
and then running it, eg.:

.. code-block:: console
   
    $ chmod +x Miniforge3-Linux-x86-64.sh
    $ ./Miniforge3-Linux-x86-64.sh

I recommend using miniforge instead of Anaconda (or any software provided by
the company of the same name); this is because, while the their software
source code is available, Anaconda's license is _not_ open-source, meaning
you might have to pay for using their software. Not only is this an antagonistic
practice to the open-source community, it makes for difficult and confusing
legal questions about whether or not you (and your institution) have to
pay for using what seems like a free software.

Follow the installation instructions, and you should end up with a working
conda installation, which you can check with:

.. code-block:: console

    $ conda --help


SLEAP environment setup
~~~~~~~~~~~~~~~~~~~~~~~

You can generally follow the directions for install from the SLEAP homepage;
I recreate them here, but if this doesn't work in the future, make sure to
check there. One thing to note is that the directions on the homepage
reference Anaconda; I have no idea if using their repository means you are
subject to their (stupid) license mentioned in the previous section, but just
to be safe, I'd like to avoid it.

.. code-block:: console

    $ conda create --name sleap pip python=3.7.12 cudatoolkit=11.3 cudnn=8.2 -c conda-forge -c nvidia
    $ conda activate sleap && pip install sleap[pypi]==1.3.3

.. note::

    If you plan on running the training or tracking on a PC with a GPU, make sure
    that GPU is available during the installation above. For example, if you
    are using a cluster/server, you need to do this on an actual compute node,
    *not* on a login node that doesn't have access to the GPU(s).

I use SLEAP v1.3.3 because there were some bugs with running newer versions
at the time. These may be fixed now, so feel free to experiment with newer
versions.

Analysis environment setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

For the rest of the analyses, we'll just use a typical scientific Python
environment. I used Python 3.11, but this choice isn't particularly important.

.. code-block:: console

    $ conda create --name analysis pip python=3.11

These packages should be installed if you install the library via ``pip``,
but just in case, you can do:

.. code-block:: console

    $ conda activate analysis
    $ pip install numpy scipy matplotlib jupyter-lab tqdm h5py
