Loading data
------------

This tutorial covers how to load and work with the processed trajectory
data from the publication.

.. list-table:: Available datasets
    :widths: 10 40 8 8
    :header-rows: 1

    * - Dataset
      - Description
      - Size
      - Link

    * - dataset_1
      - Primary dataset analyzed in the publication. Videos recorded with Nikon D800E at 60Hz. Over 100 individual ants recorded for roughly 20 minutes each.
      - 90.2 MB
      - Link

    * - dataset_2
      - Supplementary dataset. Videos recorded with Phantom v641 high speed camera. Contains only a few trajectories recorded for between 10-50 seconds each (due to limitations with high framerate).
      - 0.25 MB
      - Link


HDF5 structure
~~~~~~~~~~~~~~

All datasets are stored in the same hdf5 format, which means they can be
easily swapped out.

The basic structure is:

.. code-block:: none 

    dataset_1.h5
        |
        |- 2024-05-20_A:1
        |      |- points
        |      \- frames
        |
        |- 2024-05-20_A:2
        |      |- points
        |      \- frames
        |
       ...

``points`` will be an ``(N,2)`` array, giving the two-dimensional coordinates
of the ant throughout the segment, and ``frames`` with be a ``(N)`` array giving
the frame of the video corresponding to each point in ``points``. Since we
do interpolation within segments, ``frames`` should be a continuous (in the sense
that there are no missing values) array of integers. These give the actual
frame number of the original video, so can be used to determine the order
of segments in a video.

All metadata is attached to the ``points`` dataset within the file.

For example, if we want to access the first segment from experiment
``2024-05-20_A``, we do:

.. code-block::

    import h5py

    with h5py.File('dataset_1.h5') as f:

        trajectory = f["2024-05-20_A:1"]["points"][:]
        time = f["2024-05-20_A:1"]["frames"][:]
        metadata = dict(f["2024-05-20_A:1"]["points"].attrs)


``loadAntData``
~~~~~~~~~~~~~~~

The easiest way to work with the data is to use the :meth:`tracker.loadAntData`
function, which will return the position, time, and metadata for each segment.
This function has many options, which allow you to crop the data near the walls,
exclude outliers, exclude segments shorter than a threshold, etc.

.. code-block::

    from tracker import loadAntData

    dataFile = '/path/to/ant_dataset_1.h5'

    minimumLength = 5 # seconds
    smoothingWindow = 10 # frames
    maximumTimeSkip = 10 # frames

    excludeOutliers = True
    excludeShortSegments = False
    excludeObjects = True

    padding = None # mm
    inversePadding = None # mm

    dataArr, timeArr, metadataArr = loadAntData(dataFile, minimumLength=minimumLength,
                                                smoothingWindow=smoothingWindow, maximumTimeSkip=maximumTimeSkip,
                                                excludeOutliers=excludeOutliers, excludeShortSegments=excludeShortSegments,
                                                excludeObjects=excludeObjects,
                                                borderPadding=padding, inverseBorderPadding=inversePadding,
                                                debug=True)

Now, if I want to analyze a particular segment, I can just index the three
arrays returned by the function, eg.:

.. code-block::

    # Plot the first segment
    import matplotlib.pyplot as plt

    plt.scatter(*dataArr[0].T)
    plt.show()

We can use the metadata to look for all segments that satisfy some criteria,
eg.:

.. code-block::

    # Find all segments where the enclosure was cleaned prior to the experiment
    import numpy as np
    segmentIndices = np.where([m["cleaned"] for m in metadataArr])[0]

    # Plot the first segment that was cleaned
    plt.scatter(*dataArr[segmentIndices[0]].T)
    plt.show()


Metadata
~~~~~~~~

The table below lists all metadata included alongside the data. You can access
the values exactly as the above example checks if each segment is part of a
trial with or without cleaning.

.. list-table:: Metadata available for each segment.
    :widths: 10 40 10 10
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Example

    * - ant_identity
      - Unique identifier for the ant used in this experiment.
      - string
      - ``'2024-05-20:1'``

    * - cleaned
      - Whether the enclosure was cleaned with soap and ethanol after the previous experiment.
      - bool
      - ``True``

    * - cleaning_notes
      - Notes about the cleaning of the enclosure, if something seemed weird or otherwise notable.
      - string
      - 

    * - collection_date
      - The date on which the ant was collected from outside.
      - string
      - ``'2024-05-20'``

    * - collection_time
      - The time at which the ant was collected from outside, given in 24 hour format.
      - string
      - ``'08:00'``

    * - converted_to_mm
      - Whether the ``points`` array is converted to mm units (instead of pixels).
      - bool
      - ``True``

    * - cpu
      - Whether the SLEAP tracking was run on a CPU (True) or GPU (False).
      - bool
      - ``True``

    * - dataset
      - The name of the experiment that the segment comes from.
      - string
      - ``'2024-05-20-1A'``

    * - date
      - The date on which the experiment was performed.
      - string
      - ``'2024-05-20'``

    * - fps
      - The framerate of the experimental videos.
      - float
      - ``60``

    * - image_size
      - The image size (resolution) of the experimental videos.
      - list of int
      - ``[1280 736]``

    * - lighting
      - Whether the overhead lights were on ('On') or off ('Off') for the experiment.
      - string
      - ``'On'``

    * - mm_per_pixel
      - The size of each pixel in the experimental videos in mm (same for both dimensions).
      - float
      - ``0.198``

    * - name
      - The name of the segment.
      - string
      - ``'2024-05-20-1A:0'``

    * - nodes
      - The names of the keypoints tracked on the ant (and then averaged).
      - array of string
      - ``array(['head', 'thorax', 'abdomen'])``

    * - notes
      - General notes about the experiment and the behavior of the ant.
      - string
      -

    * - objects
      - Objects that were in the arena during the experiments, for example food or water to test behavior.
      - list of string
      - ``['water']``

    * - outlier
      - Whether the trial has been marked as an outlier; if True, see ``outlier_reason``.
      - bool
      - ``False``

    * - outlier_reason
      - Why the trial was marked as an outlier (if applicable, otherwise empty string).
      - string
      -

    * - pretreatment
      - Whether ants were kept together with other ants before the experiment ("together") or isolated ("isolated").
      - string
      - ``'together'``

    * - sleap_models
      - The model(s) used in tracking the experimental videos with SLEAP.
      - array of string
      - 

    * - sleap_version
      - The version of SLEAP used to track the videos.
      - string
      - ``'1.3.3'``

    * - temperature
      - The temperature during the experiment; recorded for most but not all experiments.
      - float
      - ``22.0``

    * - totalt_time_minutes
      - The length of the segment in minutes.
      - float
      - ``19.91``

    * - total_time_seconds
      - The length of the segment in seconds.
      - float
      - ``1194.8``

    * - video_name
      - The original name of the video from SLEAP tracking.
      - string
      - ``'2024-05-20-1A_cropped.mp4'``
