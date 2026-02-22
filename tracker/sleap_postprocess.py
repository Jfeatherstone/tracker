import numpy as np
import sleap
import os

def trackSleapFile(datasetPath, interpolationLength=10, minScore=0.4, maxDistance=np.inf):
    """
    Load in a SLEAP tracking file (.slp) and extract the trajectory
    information from it.

    This function will ignore any tracks currently present in the
    data, and generate new ones, assuming that there is only a
    single individual across the whole video. This is done using a
    very simple custom tracking algorithm:

        Add the position of the instance closest to the previous
        frame to the trajectory.

        If the previous frame(s) are missing, but there is a valid
        frame less than X frames ago, add the position closest to
        that previous one, and linearly interpolate the missing points.

        Otherwise, take the instance in the frame that has the best
        prediction score.

    I think this is a flexible algorithm that is able to interpolate to
    fill in small gaps, but doesn't introduce too many artifacts
    from interpolation. The value of X above is the param
    `interpolationLength`.

    There is really not a lot of documentation how to work directly
    with the .slp project files, so I had to really experiment
    to get this to work well. I guess the alternative is to properly
    export each project as an h5 file, but I didn't want to have
    to do that for each and every file I have.

    For more information about this function (and how it is used) see the
    notebook ``00_Preprocessing.ipynb``.
    """

    dataset = sleap.load_file(datasetPath)

    # The predicted instances are considered as labeled frames
    # so this is the main object we are working with.

    # We don't define how many positions there might be in each
    # frame yet since there could be multiple detections or
    # no detections.
    positionArr = []
    scoreArr = []

    # For now, this time is in units of frames, not seconds or
    # anything (since we don't have FPS information yet).
    timeArr = []

    for i in range(len(dataset.labeled_frames)):
        # Grab the "instances" of skeletons that were detected
        # on this frame
        instances = dataset.labeled_frames[i].instances
        instances = [inst for inst in instances if inst.score > minScore]

        if len(instances) > 0:
            # This gets the mean of all keypoints in a single instance, for
            # however many instances we have this frame.
            keypointPositions = np.array([np.mean([(p.x, p.y) for p in instances[i].points], axis=0) for i in range(len(instances))])

            positionArr.append(keypointPositions)
            scoreArr.append(np.array([inst.score for inst in instances]))
            timeArr.append(i)


    cleanedPositionArr = []
    cleanedTimeArr = []
    # Now we resolve frames that have more than one instance, and
    # interpolate small gaps
    addedFrames = 0

    for i in range(len(positionArr)):

        # If we are in the frame immediately following the previous
        # one, then we can just add the closest instance from the current
        # frame
        if (len(cleanedTimeArr) > 0) and (timeArr[i] == cleanedTimeArr[-1] + 1):
            distanceArr = np.sqrt(np.sum((cleanedPositionArr[-1] - positionArr[i])**2, axis=-1))
            if np.min(distanceArr) <= maxDistance:
                cleanedPositionArr.append(positionArr[i][np.argmin(distanceArr)])
                cleanedTimeArr.append(timeArr[i])
            # If this new instance is too far away, we just ignore it  as it's
            # probably jitter

        # If we are within some time of the previous frame, but not
        # directly after it, we need to interpolate
        elif (len(cleanedTimeArr) > 0) and (timeArr[i] - cleanedTimeArr[-1] <= interpolationLength):
            distanceArr = np.sqrt(np.sum((cleanedPositionArr[-1] - positionArr[i])**2, axis=-1))

            # Make sure we aren't too far away
            if np.min(distanceArr) <= maxDistance * (timeArr[i] - cleanedTimeArr[-1]):

                # Interpolate between the current position and the new
                # one
                finalPosition = positionArr[i][np.argmin(distanceArr)]
                interpolationSteps = timeArr[i] - cleanedTimeArr[-1]

                interpolationPointsX = np.linspace(cleanedPositionArr[-1][0], finalPosition[0], interpolationSteps+1)
                interpolationPointsY = np.linspace(cleanedPositionArr[-1][1], finalPosition[1], interpolationSteps+1)

                for j in range(interpolationSteps):
                    cleanedPositionArr.append(np.array([interpolationPointsX[j+1], interpolationPointsY[j+1]]))
                    cleanedTimeArr.append(timeArr[i] - interpolationSteps + j + 1)

        # If there has been a large gap after the previous frame,
        # or this is the first frame in the video,
        # we just take whichever instance has a better score.
        else:
            cleanedPositionArr.append(positionArr[i][np.argmax(scoreArr[i])])
            cleanedTimeArr.append(timeArr[i])


    cleanedTimeArr = np.array(cleanedTimeArr)
    cleanedPositionArr = np.array(cleanedPositionArr)

    # +1 so we cut *after* the jump happens
    splitPoints = np.where((cleanedTimeArr[1:] - cleanedTimeArr[:-1]) > 1)[0] + 1
    splitPoints = [0] + list(splitPoints) + [len(cleanedPositionArr)]

    segmentArr = []
    segmentTimeArr = []

    for i in range(1, len(splitPoints)):
        segmentArr.append(cleanedPositionArr[splitPoints[i-1]:splitPoints[i]])
        segmentTimeArr.append(cleanedTimeArr[splitPoints[i-1]:splitPoints[i]])

    metadata = dataset.to_dict()
    # We don't need these entries, as we
    # are cleaning up this same data in our other
    # return values
    del metadata["labels"]
    del metadata["tracks"]

    return segmentArr, segmentTimeArr, metadata


def cleanMetadata(metadata):
    """
    Extract manually identified important metadata from
    the SLEAP metadata.
    """
    newMetadata = {}

    # Clean up the nodes structure
    nodeNames = [n["name"] for n in metadata["nodes"]]
    newMetadata["nodes"] = nodeNames
    newMetadata["video_name"] = os.path.basename(metadata["videos"][0]["backend"]["filename"])
    newMetadata["dataset"] = os.path.basename(newMetadata["video_name"].split('_')[0])

    newMetadata["models"] = metadata["provenance"]["model_paths"]
    newMetadata["sleap_version"] = metadata["provenance"]["sleap_version"]

    newMetadata.update(metadata["provenance"]["args"])

    # Some things that I have manually marked as not important
    keysToRemove = ["frames", "output"]
    for k in keysToRemove:
        del newMetadata[k]

    # We need to convert any lists to numpy arrays
    # of strings, otherwise we can't save as an h5 file
    newMetadata["nodes"] = np.array(newMetadata["nodes"], dtype='S')
    newMetadata["models"] = np.array(newMetadata["models"], dtype='S')

    # Similarly, we can't have None values, so we just convert them
    # to strings (that data is pretty much read only anyway, so it
    # doesn't really need to be exactly the correct type)
    for k,v in newMetadata.items():
        if type(v) is type(None):
            newMetadata[k] = "None"

    return newMetadata


# Old Method, not recommended to use anymore
def _extractTrackingData(datasetPath):
    """
    Load in a SLEAP tracking file (.slp) and extract the trajectory
    information from it.

    There is really not a lot of documentation how to work directly
    with the .slp project files, so I had to really experiment
    to get this to work well. I guess the alternative is to properly
    export each project as an h5 file, but I didn't want to have
    to do that for each and every file I have.
    """

    dataset = sleap.load_file(datasetPath)

    # The predicted instances are considered as labeled frames
    # so this is the main object we are working with.

    # The first thing we want to do is extract to which "track"
    # each predicted frame belongs to.
    trackIdentityArr = np.zeros(len(dataset.labeled_frames)) - 1
    positionArr = np.zeros((len(dataset.labeled_frames), 2))
    # Start as nan
    positionArr[:] = np.nan

    uniqueTracks = np.sort(np.unique([t.name for t in dataset.tracks]))
    nameConversionDict = dict(zip(uniqueTracks, np.arange(len(uniqueTracks))))

    for i in range(len(dataset.labeled_frames)):
        # Grab the "instances" of skeletons that were detected
        # on this frame
        instances = dataset.labeled_frames[i].instances

        # Make sure we have exactly one ant
        if len(instances) != 1:
            continue

        # Grab the name of the "track" (continuous trajectory) and convert
        # it to an index
        track = instances[0].track.name
        trackIdentityArr[i] = nameConversionDict.get(track, -1)

        # Grab the positions of the detected points and average them
        keypointPositions = np.array([(p.x, p.y) for p in instances[0].points])
        positionArr[i] = np.mean(keypointPositions, axis=0)

    # Now we group the tracks together into continuous trajectories
    trajectoryList = []
    frameList = []

    for i in range(len(uniqueTracks)):
        trackIndices = np.where(trackIdentityArr == i)

        trajectoryList.append(positionArr[trackIndices])
        frameList.append(trackIndices[0])

    metadata = dataset.to_dict()

    # We don't need these entries, as we
    # are cleaning up this same data in our other
    # return values
    del metadata["labels"]
    del metadata["tracks"]

    return trajectoryList, frameList, metadata


#     # Grab the metadata for the tracking
#     metadata = dataset.to_dict()

#     # We don't need these entries, as we
#     # are cleaning up this same data in our other
#     # return values
#     del metadata["labels"]
#     del metadata["tracks"]

#     # The unique names of tracks
#     uniqueTrackNameArr = [t.name for t in dataset.tracks]
#     # The name of the track that each frame belongs to
#     trackNameArr = np.array([instance.track.name for instance in dataset.predicted_instances])

#     # The frame number of the original video
#     frameIndexArr = np.array([instance.frame_idx for instance in dataset.predicted_instances])

#     # The names of each part that is tracked on the ant
#     partLabels = [n.name for n in dataset.skeleton.nodes]

#     # The position of each tracked part for each frame
#     # Slightly more complex, since the format is a little weird,
#     # and we have to account for the possibility that not all
#     # keypoints were tracked.
#     nodeArr = [instance.nodes_points for instance in dataset.predicted_instances]

#     # This is the names of nodes tracked for each frame
#     # Most entries should be the same as partLabels, but
#     # some might be missing.
#     nodeNameArr = [[n[0].name for n in tuple(item)] for item in nodeArr]
#     # Positions of all tracked parts
#     pointArr = [np.array([(n[1].x, n[1].y) for n in tuple(item)]) for item in nodeArr]

#     # Throw away frames where we are missing some keypoints
#     goodIndices = np.where(np.array([len(p) for p in pointArr]) == len(partLabels))

#     # Take only the fully tracked frames
#     # Have to delete indices like this, since we can't
#     # create a numpy array (that isn't of object type)
#     # since we might have 2 or 3 points at each frame.
#     for i in range(len(pointArr))[::-1]:
#         if len(pointArr[i]) != len(partLabels):
#             del pointArr[i]

#     pointArr = np.array(pointArr)
#     trackNameArr = trackNameArr[goodIndices]
#     frameIndexArr = frameIndexArr[goodIndices]

#     # Now we group the tracks together into continuous trajectories
#     trajectoryList = []
#     frameList = []

#     plt.plot(*pointArr.T)
#     plt.show()

#     for i in range(len(uniqueTrackNameArr)):
#         trackIndices = np.where(trackNameArr == uniqueTrackNameArr[i])

#         trajectoryList.append(pointArr[trackIndices])
#         frameList.append(frameIndexArr[trackIndices])

#     return trajectoryList, frameList, metadata


