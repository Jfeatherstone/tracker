import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def discretizeTrajectoryIndices(trajectoryArr, c=1, velocityThreshold=1, dt=1, distanceThreshold=0.05, minSteps=1, minDistancePerRun=0, ):
    """
    Discretize a trajectory as if it were a run-and-tumble process,
    with waiting times before each tumble. Return the indices of
    the run or waiting time to which each point in the original trajectory
    belongs.

    Trajectories are first segmented by finding regions where the
    trajectory has very low velocity (so the process is probably
    standing still). Then, each of these segments is split based
    on orientation to get relatively straight runs. Afterwards, we do
    some clean up to make sure the detected runs actually seem like runs.

    The process of discretizing a continuous trajectory (even if it is
    discretly sampled) is very subjective, and care should be taken
    to justify the choices of parameters for this discretization
    algorithm.
    
    Parameters
    ----------
    trajectoryArr : numpy.ndarray[N,2]
        The points along the trajectory.

    c : float
        The colinearity threshold used to decide where to split the
        trajectory to form straight runs. The colinearity is defined
        as the dot product between two unit vectors, so can take a
        value in the range [-1, 1].

        Note that any new step of the trajectory is compared against
        the *mean* direction vector of the current run, to decide if
        it is added to that run or if it should start a new one. Some
        works often just compare the new direction vector to the previous
        one, but this is more susceptible to give unrepresentative
        discretizations in the presence of noise. Some works refer to this
        as a "nonlocal" technique [eg. 1].

        If you'd rather define the threshold in terms of an angle instead
        of a scalar, your colinearity threshold will be:

            cos(theta)

        where theta is the critical angle.

    velocityThreshold : float
        The velocity threshold used to identify when the process is waiting,
        or not moving. The velocity is slightly smoothed before being
        compared to the threshold, to avoid missing waiting times because
        of instantaneous jitter in the middle of an otherwise stationary
        period.

    dt : float
        The time interval between steps in the original trajectory. Used
        to calculate the velocity.

    distanceThreshold : float
        The distance under which a step won't be considered to go against the
        current run; useful if your data is noisy.

    minSteps : int
        The minimum number of steps that must be going in roughly the same
        direction to constitute a run.

    minDistancePerRun : float
        The minimum distance a run must measure from beginning point to 
        end point in order for it to be considered an actual run and not
        just a wait time that happens to be drifting.


    Returns
    -------
    runIntervals : numpy.ndarray[M,2]
        The indices of the start ([:,0]) and end ([:,1]) of each discrete
        run.

    waitingTimes : numpy.ndarray[L,2]
        The indices of the start ([:,0]) and end ([:,1]) of each discrete
        waiting time.

    References
    ----------
    [1] Reynolds, A. M., Smith, A. D., Menzel, R., Greggers, U., Reynolds,
    D. R., & Riley, J. R. (2007). Displaced Honey Bees Perform Optimal
    Scale-Free Search Flights. Ecology, 88(8), 1955–1961.
    https://doi.org/10.1890/06-1916.1

    """

    angleThreshold = np.arccos(c)

    # This array identifies which "segment" a point belongs to, which are defined
    # by velocity thresholding
    # Value of -1 means it has not been assigned yet
    segmentIdentityArr = np.zeros(len(trajectoryArr)) - 1

    # This array identifies which "run" a point belongs to, which are defined
    # by orientation thresholding (within each segment)
    # Value of -1 means it has not been assigned yet
    runIdentityArr = np.zeros(len(trajectoryArr)) - 1

    # We will need the step sizes (in the measured frame)
    # to weight the contributions to the overall run direction
    originalStepDirectionArr = trajectoryArr[1:] - trajectoryArr[:-1]
    originalStepSizeArr = np.sqrt(np.sum((trajectoryArr[1:] - trajectoryArr[:-1])**2, axis=-1))
    # This will have length one less than trajectory arr, so we add a zero value at the beginning
    # so it can be indexed the same.
    originalStepSizeArr = np.concatenate((np.zeros(1), originalStepSizeArr))
    # Note that the step sizes also double as the velocity (we just add
    # a factor of dt, which is optional, so potentially could be exactly the same)
    velocityArr = originalStepSizeArr / dt

    # That being said, we do smooth the velocity slightly since it
    # helps with identfying regions with average low velocity
    velocityArr = savgol_filter(velocityArr, 5, 1)
    
    # First, we need to identify segments based on velocity thresholding.
    segmentIdentityArr[0] = 0
    currentCount = 0
    for i in range(1, len(trajectoryArr)):
        # We need minSteps points in a row that have velocity
        # lower than the threshold
        if velocityArr[i] < velocityThreshold:
            currentCount += 1
        else:
            currentCount = 0

        if currentCount == minSteps:
            # Add a new segment, and change the last minSteps entries since
            # they were actually a part of this segment.
            for j in range(minSteps):
                segmentIdentityArr[i-j] = segmentIdentityArr[i-minSteps-1] + 1
                
        else:
            # Continue the current segment
            segmentIdentityArr[i] = segmentIdentityArr[i-1]

    # DEBUG
    plt.plot(segmentIdentityArr)
    plt.show()

    plt.plot(velocityArr)
    plt.show()

    plt.plot(trajectoryArr[:,0])
    plt.plot(trajectoryArr[:,1])
    plt.show()

    absoluteAngleArr = np.arctan2(*(originalStepDirectionArr.T / np.sqrt(np.sum(originalStepDirectionArr**2, axis=-1))))
    absoluteAngleArr[velocityArr[1:] < velocityThreshold] = np.nan
    plt.plot(absoluteAngleArr, 'o')
    plt.show()

    # Now we want to identify runs within each segment by using the angle
    # threshold.
    # We first should identify the intervals of each segment, from our
    # list of which segment each point belongs to
    segmentIntervals = np.array([[np.min(np.where(segmentIdentityArr == i)[0]), np.max(np.where(segmentIdentityArr == i)[0])] for i in np.unique(segmentIdentityArr)])
    # This above list has entries [[start_1, end_1], [start_2, end_2], etc.]
    
    runIdentityArr[0] = 0
    for i in range(len(segmentIntervals)):
        startIndex = max(1, segmentIntervals[i][0])
        endIndex = min(len(trajectoryArr), segmentIntervals[i][1])
        
        # Iterate over the indices that are associated with this segment
        for j in range(startIndex, endIndex):
            # Check if this point fits in with the current run
            currentRunIndices = np.where(runIdentityArr == runIdentityArr[j-1])[0]
            
            # Compute the average direction the current run is going
            currentRunDirection = np.average(originalStepDirectionArr[currentRunIndices - 1],
                                             axis=0)
    
            currentRunDirection /= np.sqrt(np.sum(currentRunDirection**2))
            currentStepDirection = originalStepDirectionArr[j-1] / originalStepSizeArr[j]
    
            # Now compute the difference in angle between the current direction
            # and the current run
            angleDiff = np.dot(currentStepDirection, currentRunDirection)
    
            # If we are within the threshold, this point joins the current
            # run.
            if angleDiff >= angleThreshold or originalStepSizeArr[j] <= distanceThreshold:
                runIdentityArr[j] = runIdentityArr[j-1]
            # Otherwise we start a new run
            else:
                runIdentityArr[j] = np.max(runIdentityArr) + 1
            
        if runIdentityArr[endIndex] == -1:
            runIdentityArr[endIndex] = np.max(runIdentityArr)
    # plt.plot(segmentIdentityArr)
    # plt.plot(runIdentityArr)
    # plt.show()

    # Now we compute the length of each run (time, not space)
    # by counting how many steps are included
    uniqueRuns, runStepsArr = np.unique(runIdentityArr, return_counts=True)
        
    # Now we have to compute waiting times between runs.
    # Any run that is shorter than the threshold minStepsPerRun
    # is considered to be a waiting time.
    waitingTimeIntervals = []
    properRunIntervals = []
    inWaitingTime = False
    
    for i in range(len(uniqueRuns)):
        # To see if this is a valid run or not, we need to
        # check two conditions:
        # 1: that the run contains more than minStepsPerRun discrete
        # steps.
        # 2. that the run travels a distance ( |end - start| distance)
        # more than minDistancePerRun.
        originalIndices = np.where(runIdentityArr == uniqueRuns[i])[0]

        runLength = np.sqrt(np.sum((trajectoryArr[np.max(originalIndices)] - trajectoryArr[np.min(originalIndices)])**2))

        if runStepsArr[i] >= minSteps and runLength >= minDistancePerRun:
            inWaitingTime = False
            
            # Save the indices (of the original set of points, trajectoryArr)
            # of the start and end of the run
            properRunIntervals += [[np.min(originalIndices),
                                    np.max(originalIndices)]]
            continue

        # If the previous run(s) were also waiting times,
        # we just continue that one
        if inWaitingTime:
            waitingTimeIntervals[-1][1] = np.max(np.where(runIdentityArr == uniqueRuns[i])[0])
            
        # Otherwise, we start a new waiting time interval
        else:
            waitingTimeIntervals += [[np.min(np.where(runIdentityArr == uniqueRuns[i])[0]),
                                      np.max(np.where(runIdentityArr == uniqueRuns[i])[0])]]
            inWaitingTime = True

    if len(properRunIntervals) == 0:
        return np.zeros(0), np.zeros(0)
        
    # Since we may have merged some waiting times that will form a new run,
    # we need to check that again
    indicesToSwitch = []
    for i in range(len(waitingTimeIntervals)):
        # If we move more than the minimum distance during a
        # (merged) waiting time, probably this isn't actually a waiting time
        driftLength = np.sqrt(np.sum((trajectoryArr[waitingTimeIntervals[i][0]] - trajectoryArr[waitingTimeIntervals[i][1]])**2))
        if driftLength >= minDistancePerRun:
            indicesToSwitch += [i]

    # Switch those intervals to the run list and remove
    # them from the waiting time one
    properRunIntervals += [waitingTimeIntervals[i] for i in indicesToSwitch]
    waitingTimeIntervals = [waitingTimeIntervals[i] for i in range(len(waitingTimeIntervals)) if i not in indicesToSwitch] 

    # Sort since we just added some new runs out of order
    order = np.argsort(np.array(properRunIntervals)[:,0])
    properRunIntervals = np.array(properRunIntervals)[order]
    
    # Now we need to fill in the waiting time intervals that might
    # be very short and therefore didn't show up in the list
    # waitingTimeIntervals
    for i in range(1, len(properRunIntervals)):
        if properRunIntervals[i][0] == properRunIntervals[i-1][1] + 1:
            waitingTimeIntervals += [[properRunIntervals[i][0] - 1, properRunIntervals[i][0]]]
            properRunIntervals[i-1][1] += 1

    # Sort since we just added some new waiting times out of order
    if len(waitingTimeIntervals) > 0:
        order = np.argsort(np.array(waitingTimeIntervals)[:,0])
        waitingTimeIntervals = np.array(waitingTimeIntervals)[order]

    # for i in range(len(waitingTimeIntervals)):
    #     plt.scatter(*trajectoryArr[waitingTimeIntervals[i][0]:waitingTimeIntervals[i][1]].T, c='tab:orange', alpha=0.2)

    # for i in range(len(properRunIntervals)):
    #     plt.scatter(*trajectoryArr[properRunIntervals[i][0]:properRunIntervals[i][1]].T, c='tab:blue', alpha=0.2)

    # plt.show()
    return properRunIntervals, waitingTimeIntervals


def discretizeTrajectory(trajectoryArr, c=1, velocityThreshold=1, dt=1, distanceThreshold=0.05, minSteps=1, minDistancePerRun=0):
    """
    Discretize a trajectory into straight runs and waiting times.

    See also `discretizeTrajectoryIndices()` for more information about
    the discretization process.

    The trajectory is constructed by taking the average between the
    start and end of successive trajectories; this is required since the
    trajectory might drift slightly during the wait time.
    
    Parameters
    ----------
    trajectoryArr : numpy.ndarray[N,2]
        The points along the trajectory.

    c : float
        The colinearity threshold used to decide where to split the
        trajectory to form straight runs. The colinearity is defined
        as the dot product between two unit vectors, so can take a
        value in the range [-1, 1].

        Note that any new step of the trajectory is compared against
        the *mean* direction vector of the current run, to decide if
        it is added to that run or if it should start a new one. Some
        works often just compare the new direction vector to the previous
        one, but this is more susceptible to give unrepresentative
        discretizations in the presence of noise. Some works refer to this
        as a "nonlocal" technique [eg. 1].

        If you'd rather define the threshold in terms of an angle instead
        of a scalar, your colinearity threshold will be:

            cos(theta)

        where theta is the critical angle.

    velocityThreshold : float
        The velocity threshold used to identify when the process is waiting,
        or not moving. The velocity is slightly smoothed before being
        compared to the threshold, to avoid missing waiting times because
        of instantaneous jitter in the middle of an otherwise stationary
        period.

    dt : float
        The time interval between steps in the original trajectory. Used
        to calculate the velocity.

    distanceThreshold : float
        The distance under which a step won't be considered to go against the
        current run; useful if your data is noisy.

    minSteps : int
        The minimum number of steps that must be going in roughly the same
        direction to constitute a run.

    minDistancePerRun : float
        The minimum distance a run must measure from beginning point to 
        end point in order for it to be considered an actual run and not
        just a wait time that happens to be drifting.


    Returns
    -------
    discreteTrajectoryArr : numpy.ndarray[M,2]
        The new points of the discretized trajectory.

    waitingTimeArr : numpy.ndarray[M]
        The time spent waiting in between each run.

    runTimeArr : numpy.ndarray[M-1]
        The time spent moving during each run.

    References
    ----------
    [1] Reynolds, A. M., Smith, A. D., Menzel, R., Greggers, U., Reynolds,
    D. R., & Riley, J. R. (2007). Displaced Honey Bees Perform Optimal
    Scale-Free Search Flights. Ecology, 88(8), 1955–1961.
    https://doi.org/10.1890/06-1916.1
    """
    # Get the indices of intervals for runs and waits
    properRunIntervals, waitingTimeIntervals = discretizeTrajectoryIndices(trajectoryArr,
                                                                           c,
                                                                           velocityThreshold,
                                                                           dt,
                                                                           distanceThreshold,
                                                                           minSteps,
                                                                           minDistancePerRun)

    
    if len(properRunIntervals) == 0 or len(waitingTimeIntervals) == 0:
        return np.array(trajectoryArr)[[0, -1]], np.array([len(trajectoryArr)]), np.zeros(1)
        
    # Compute the start and end of the runs as the mean between the end of
    # the previous run and the start of the current run.

    # First point is just the first point in the first run interval (no
    # averaging)
    discretePoints = [trajectoryArr[properRunIntervals[0][0]]]
    # Now we take the average of the end of the previous run and the start
    # of the new one.
    for i in range(1, len(properRunIntervals)-1):
        discretePoints += [np.mean([trajectoryArr[properRunIntervals[i][0]],
                                    trajectoryArr[properRunIntervals[i-1][1]]], axis=0)]

    # And we add the last point of the last run
    discretePoints += [trajectoryArr[properRunIntervals[-1][1]]]

    # If we had L run intervals, we should now have L+1 discrete points
    discretePoints = np.array(discretePoints)

    waitingTimes = waitingTimeIntervals[:,1] - waitingTimeIntervals[:,0]
    runTimes = (properRunIntervals[:,1] - properRunIntervals[:,0]) * dt

    # print(len(waitingTimes), len(runTimes), len(discretePoints))
    # print(properRunIntervals)
    # print(waitingTimeIntervals)
    
    return discretePoints, waitingTimes, runTimes
