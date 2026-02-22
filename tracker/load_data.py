"""
This file contains the preferred method to load in the preprocessed
ant tracking data. This can be applied to either dataset included
in this work.
"""
import numpy as np

import h5py

from tqdm import tqdm
from scipy.signal import savgol_filter

def loadAntData(inputPath,
                minimumLength=10,
                smoothingWindow=10,
                maximumTimeSkip=10,
                excludeOutliers=True,
                excludeShortSegments=True,
                excludeObjects=False,
                borderPadding=None,
                inverseBorderPadding=None,
                debug=False):
    """
    Load in ant trajectory data, preprocessed by the file ``00_PreprocessData.ipynb``.
    
    While the SLEAP model that tracks the videos has multiple keypoints,
    the preprocessing averages those three keypoints to get the center of
    mass over time. You can explore the individual keypoint trajectories
    if you would like using the raw sleap files, but they are much more
    susceptible to discretization and jitter.
    
    By default, the trajectory is smoothed as discussed in the manuscript.
    This can be adjusted if you'd like with the ``smoothingWindow`` parameter.

    Gaps in the trajectory are also removed; this can be adjusted by
    changing the ``maximumTimeSkip`` parameter.

    You can remove parts of the trajectory that are close to the boundary
    as desired using the ``borderPadding`` or ``inverseBorderPadding`` (for
    only keeping data around the boundary). The boundary is computed for
    each individual trajectory.

    If part of a segment goes near the boundary, that segment will be
    split into multiple pieces to avoid having a large gap(s) in the middle.

    Parameters
    ----------
    inputPath : str
        Path to the input h5 file.
        
    minimumLength : float
        The minimum length of a segment (in seconds) that will
        be included in the returned list of segments.
        
        Only has an effect if `excludeShortSegments=True`.

        Note that this value is given in *seconds*, or whatever the native
        unit of time for the trajectory is.
        
    smoothingWindow : int
        The size of the smoothing window applied to the trajectory
        data.

        Note that this value is given in *frames* (or steps, indices, etc.),
        not the native time unit of the trajectories.

    maximumTimeSkip : int
        The maximum amount of frames that are allowed to be skipped
        without breaking up a trajectory into multiple pieces. Note
        that this value is given in *frames* (or steps, indices, etc.),
        not the native time unit of the trajectories.
        
    excludeOutliers : bool
        Whether to exclude trials annotated as an outlier during the
        preprocessing. For more information on why trials might be
        annotated as an outlier, see file `00_PreprocessData.ipynb`.
        
    excludeShortSegments : bool
        Whether to apply the provided value of `minimumLength` as a
        cutoff for the length of segments.

    excludeObjects : bool
        Whether to exclude trials that had an object in the enclosure, 
        including food, water, etc.
        
    borderPadding : float, optional
        The size of padding around the borders of the arena to ignore.
        If a segment is partly contained in this region, only the points
        which actually fall in the region are deleted.
        
        Be careful when calculating speeds, angle turns, etc., as you will
        have to make sure to ignore the times when the process jumps. The
        array `timeArr` can be used to determine when this is necessary.
        
    inverseBorderPadding : float, optional
        The size of padding around the borders of the arena to keep.
        Any points that are part of a segment that deviate from this
        region will be deleted, keeping only the trajectories near the
        boundary.

        Be careful when calculating speeds, angle turns, etc., as you will
        have to make sure to ignore the times when the process jumps. The
        array `timeArr` can be used to determine when this is necessary.

    debug : bool
        Whether to show progress bars during the loading process.
        
    Returns
    -------
    dataArr : list of np.array[N_i, 2]
        The list of positions of the ant throughout time for each segment.
        
    timeArr : list of np.array[N_i]
        The list of time points at which each position in dataArr[i] is
        sampled. The first data point is the actual time within
        the original video at which this trajectory starts, and thus can
        be used to determine the order of segments within a trial.
        
    metadataArr : list of dict
        The dictionaries containing metadata information about the tracking
        procedure, experimental conditions, and preprocessing steps for
        each segment.
    """
    
    dataArr = []
    timeArr = []
    metadataArr = []

    with h5py.File(inputPath) as f:
        for i,k in tqdm(enumerate(f.keys()), total=len(list(f.keys())), desc='Loading SLEAP data') if debug else enumerate(f.keys()):
            points = f[k]["points"][:]
            frames = f[k]["frames"][:]
            metadata = dict(f[k]["points"].attrs)

            # Exclude outliers
            if excludeOutliers and metadata["outlier"]:
                continue

            # Exclude short segments
            if excludeShortSegments and len(points) / metadata["fps"] < minimumLength:
                continue

            # We don't need to average over all keypoints
            # since this is done during preprocessing
            
            # Apply smoothing
            if len(points) > smoothingWindow and smoothingWindow > 0:
                # In general, the savgol_filter can do much more than simply 
                # windowed averaging, but here we just use it to first order,
                # which is equivalent to linearly re-interpolating the points.
                points = savgol_filter(points, smoothingWindow, 1, axis=0)
            dataArr.append(points)

            # Compute the time for each point in seconds (not frames)
            time = frames / metadata["fps"]
            timeArr.append(time)

            metadataArr.append(metadata)
            
    # Make sure we don't have request to remove the border and only keep the
    # border simultaneously.
    if (borderPadding is not None and borderPadding > 0) and (inverseBorderPadding is not None and inverseBorderPadding > 0):
        raise Exception('Can\'t satisfy values for both `borderPadding` and `inverseBorderPadding`. Please only provide one.')
        
    # If requested, we remove parts around the borders
    if borderPadding is not None and borderPadding > 0:
            
        for i in tqdm(range(len(dataArr)), desc='Removing trajectories near walls') if debug else range(len(dataArr)):
            lowBadIndices = np.where((dataArr[i] - np.min(dataArr[i], axis=0)) < borderPadding)[0]
            highBadIndices = np.where(dataArr[i] > (np.max(dataArr[i], axis=0) - borderPadding))[0]
            badIndices = np.unique(np.concatenate((lowBadIndices, highBadIndices)))
            goodIndices = [i for i in range(len(dataArr[i])) if not i in badIndices]

            dataArr[i] = dataArr[i][goodIndices]
            timeArr[i] = timeArr[i][goodIndices]
            
    # If requested, we remove parts far from the borders
    if inverseBorderPadding is not None and inverseBorderPadding > 0:

        for i in tqdm(range(len(dataArr)), desc='Keeping only trajectories near walls') if debug else range(len(dataArr)):
            lowGoodIndices = np.where((dataArr[i] - np.min(dataArr[i], axis=0)) < inverseBorderPadding)[0]
            highGoodIndices = np.where(dataArr[i] > (np.max(dataArr[i], axis=0) - inverseBorderPadding))[0]
            goodIndices = np.unique(np.concatenate((lowGoodIndices, highGoodIndices)))

            dataArr[i] = dataArr[i][goodIndices]
            timeArr[i] = timeArr[i][goodIndices]

    # If we've cropped the data (or even if the input data is weird) we
    # have to make sure that we don't have big jumps in the trajectories.
    # To fix this, we split any segment that has a time gap larger than
    # maximumTimeSkip into multiple segments.
    newDataArr = []
    newTimeArr = []
    newMetadataArr = []
    for i in range(len(dataArr)):
        timeSkipArr = timeArr[i][1:] - timeArr[i][:-1]
  
        # +1 because this finds indices for timeSkipArr, which has one less
        # entry than dataArr[i]
        breakPoints = np.where(timeSkipArr > maximumTimeSkip / metadataArr[i]["fps"])[0] + 1

        # Add the beginning and end of the trajectory to the break
        # points so we can treat all cases (even with no break in between)
        # the same
        breakPoints = [0] + list(breakPoints) + [len(dataArr[i])]
        
        # We split up the segment as necessary (even if there are not breaks)
        # 1 less than the number of actual points in the list, since we
        # added the beginning and end.
        for j in range(len(breakPoints) - 1):
            newDataArr.append(dataArr[i][breakPoints[j]:breakPoints[j+1]])
            newTimeArr.append(timeArr[i][breakPoints[j]:breakPoints[j+1]])
            newMetadataArr.append(metadataArr[i])

    # Sort by longest to shortest
    order = np.argsort([len(t) for t in newTimeArr])[::-1]
    # Remove short (empty) trials
    # Note that minimum length is given in seconds, not frames
    if excludeShortSegments:
        order = [i for i in order if len(newDataArr[i]) > minimumLength * newMetadataArr[i]["fps"]]
    else:
        # Otherwise, we just make sure it is longer than 0 length
        order = [i for i in order if len(newDataArr[i]) > 0]
    
    newTimeArr = [newTimeArr[i] for i in order]
    newMetadataArr = [newMetadataArr[i] for i in order]
    newDataArr = [newDataArr[i] for i in order]

    # If desired, remove trials that have objects in the arena.
    if excludeObjects:
        goodIndices = [i for i in range(len(newDataArr)) if newMetadataArr[i]["objects"] == '[]']

        newTimeArr = [newTimeArr[i] for i in goodIndices]
        newMetadataArr = [newMetadataArr[i] for i in goodIndices]
        newDataArr = [newDataArr[i] for i in goodIndices]

    if debug:
        print(f'Loaded {len(newDataArr)} segments!')

    return newDataArr, newTimeArr, newMetadataArr
