import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def discretizeTrajectoryIndices(trajectoryArr, c=1, velocityThreshold=1, dt=1, minSteps=1, minDistancePerRun=0, debug=False):
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

            c = cos(theta)

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

    minSteps : int
        The minimum number of steps that must be going in roughly the same
        direction to constitute a run.

    minDistancePerRun : float
        The minimum distance a run must measure from beginning point to 
        end point in order for it to be considered an actual run and not
        just a wait time that happens to be drifting.

    debug : bool
        Whether to plot debug information; helpful to deciding on good
        parameter values.


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
    # Value of nan means it has not been assigned yet
    segmentIdentityArr = np.zeros(len(trajectoryArr))
    segmentIdentityArr[:] = np.nan

    # This array identifies which "run" a point belongs to, which are defined
    # by orientation thresholding (within each segment)
    # Value of nan means it has not been assigned yet
    runIdentityArr = np.zeros(len(trajectoryArr))
    runIdentityArr[:] = np.nan

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
   
    # Compute the orientation (angle) of the walker at each step.
    absoluteAngleArr = np.arctan2(*(originalStepDirectionArr.T / np.sqrt(np.sum(originalStepDirectionArr**2, axis=-1))))

    # This array identifies whether points are waiting times or run times.
    isWaitingTimeArr = (velocityArr < velocityThreshold).astype(bool)

    # First, we need to identify segments based on velocity thresholding.
    # This means assigning each point in the trajectory to a segment, and
    # deciding if it is a waiting time or a run.
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
            previousSegmentIndex = segmentIdentityArr[i - minSteps - 1]
            # If this is the first segment, this index will be nan, so we
            # should use 0 instead
            if np.isnan(previousSegmentIndex):
                previousSegmentIndex = -1

            for j in range(minSteps):
                segmentIdentityArr[i-j] = previousSegmentIndex + 1
                
        else:
            # Continue the current segment
            segmentIdentityArr[i] = segmentIdentityArr[i-1]

    # Since picking parameters for this method should be done with care,
    # we have very detailed debug plots.
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(8.5,3.5))
        # Plot the velocity
        ax[0].scatter(np.where(velocityArr < velocityThreshold)[0],
                      velocityArr[velocityArr <= velocityThreshold], s=5, c='tab:orange')
        ax[0].scatter(np.where(velocityArr >= velocityThreshold)[0],
                      velocityArr[velocityArr >= velocityThreshold], s=5, c='tab:blue')

        ax[0].axhline(velocityThreshold, c='gray', linestyle='--')
        ax[0].set_title('Velocity')

        # Plot the absolute angle over time, separating when the velocity
        # is above or below the threshold
        ax[1].plot(np.where(velocityArr[1:] <= velocityThreshold)[0],
                 absoluteAngleArr[velocityArr[1:] <= velocityThreshold], 'o', c='tab:orange', label='Below velocity threshold', alpha=0.5)
        ax[1].plot(np.where(velocityArr[1:] > velocityThreshold)[0],
                 absoluteAngleArr[velocityArr[1:] > velocityThreshold], 'o', c='tab:blue', label='Above velocity threshold')

        #ax[1].legend()
        # Plot the segment identities as well
        dualAx = ax[1].twinx()
        dualAx.plot(segmentIdentityArr, c='black')

        ax[0].set_xlabel('Time')
        ax[1].set_xlabel('Time')
        ax[1].set_title('Absolute Angle')

        plt.show()


    # Now we want to identify runs within each segment by using the angle
    # threshold.
    # We first should identify the intervals of each segment, from our
    # list of which segment each point belongs to
    segmentIntervals = np.array([[np.min(np.where(segmentIdentityArr == i)[0]), np.max(np.where(segmentIdentityArr == i)[0])] for i in np.unique(segmentIdentityArr)])
    # This above list has entries [[start_1, end_1], [start_2, end_2], etc.]
   
    runIdentityArr[0] = 0
    for i in range(len(segmentIntervals)):
        # We need to look at the previous point in identifying the runs,
        # so we need to start 1 (if we don't already)
        startIndex = max(1, segmentIntervals[i][0])
        endIndex = min(len(trajectoryArr), segmentIntervals[i][1])
        
        # Iterate over the indices that are associated with this segment
        # We need the +1 on the second value because python's range(x,y) only
        # goes up to y-1
        for j in range(startIndex, endIndex+1):
            if isWaitingTimeArr[j]:
                continue

            # Check if this point fits in with the current run
            currentRunIndices = np.where(runIdentityArr == runIdentityArr[j-1])[0]
          
            # If there are no points in the current run, ie. this is the
            # first one, then we of course have to start a new run.
            if len(currentRunIndices) == 0:
                runIdentityArr[j] = np.nanmax(runIdentityArr) + 1
                continue

            # Compute the average direction the current run is going
            # -1 because originalStepDirectionArr has a length of 1 less than
            # the actual trajectory (or runIdentityArr) so we need to 
            # subtract 1 to index the same entry.
            currentRunDirection = np.nanmean(originalStepDirectionArr[currentRunIndices - 1],
                                             axis=0)
    
            currentRunDirection /= np.sqrt(np.sum(currentRunDirection**2))
            currentStepDirection = originalStepDirectionArr[j-1] / originalStepSizeArr[j]
    
            # Now compute the difference in angle between the current direction
            # and the current run
            angleDiff = np.dot(currentStepDirection, currentRunDirection)
    
            # If we are above the colinearity threshold, this point joins
            # the current run.
            if angleDiff >= c:
                runIdentityArr[j] = runIdentityArr[j-1]
            # Otherwise we start a new run.
            else:
                runIdentityArr[j] = np.nanmax(runIdentityArr) + 1
            
        if runIdentityArr[endIndex] == -1:
            runIdentityArr[endIndex] = np.nanmax(runIdentityArr)

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

        # These are the indices in the original trajectoryArr that are
        # part of this run.
        originalIndices = np.where(runIdentityArr == uniqueRuns[i])[0]

        # Make sure we have enough points to constitute a run.
        if len(originalIndices) >= minSteps:
            startIndex = np.nanmin(originalIndices)
            endIndex = np.nanmax(originalIndices)

            runLength = np.sqrt(np.sum((trajectoryArr[endIndex] - trajectoryArr[startIndex])**2))
            # Make sure we travel enough distance to constitute a run.
            if runLength >= minDistancePerRun:
                inWaitingTime = False
                
                # Save the indices (of the original set of points, trajectoryArr)
                # of the start and end of the run
                properRunIntervals += [[startIndex, endIndex]]

        # Otherwise, we are in a waiting time, which we will organize
        # after we've identified all of the runs.

    if len(properRunIntervals) == 0:
        return np.zeros(0), np.zeros(0)


    # Add the waiting times in
    waitingTimeIntervals = [[properRunIntervals[i][1], properRunIntervals[i+1][0]] for i in range(len(properRunIntervals)-1)]

    # Add a waiting time at the beginning if the first run doesn't start
    # immediately
    if properRunIntervals[0][0] > 0:
        waitingTimeIntervals = [[0, properRunIntervals[0][0]]] + waitingTimeIntervals

    # Add a waiting time at the end if the last run doesn't end just at the
    # end of the real trajectory
    if properRunIntervals[-1][1] < len(trajectoryArr)-1:
        waitingTimeIntervals = waitingTimeIntervals + [[properRunIntervals[-1][1], len(trajectoryArr) - 1]]

    # Sort since we just added some new runs out of order
    order = np.argsort(np.array(properRunIntervals)[:,0])
    properRunIntervals = np.array(properRunIntervals)[order]

    # This plots the individual runs within the trajectory, we can get
    # a good idea of how finely we are chopping up the trajectory.
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(8.5,3.5))

        ax[0].scatter(*trajectoryArr.T, s=5, c='black')

        for i in range(len(uniqueRuns)):
            ax[0].scatter(*trajectoryArr[np.where(runIdentityArr == uniqueRuns[i])].T, alpha=0.4)

        # Original trajectory
        ax[1].scatter(*trajectoryArr.T, s=5, c='black', alpha=0.2)

        # Runs
        for i in range(len(properRunIntervals)):
            ax[1].plot(*trajectoryArr[properRunIntervals[i]].T, linewidth=2)
      
        # Waiting times
        for i in range(len(waitingTimeIntervals)):
            ax[1].scatter(*trajectoryArr[waitingTimeIntervals[i][0]:waitingTimeIntervals[i][1]].T, alpha=0.5)

        ax[0].set_title('Initial run partitioning')
        ax[1].set_title('Run partitioning with waiting times')
        plt.show()


    waitingTimeIntervals = np.array(waitingTimeIntervals, dtype=np.int64)
    properRunIntervals = np.array(properRunIntervals, dtype=np.int64)

    # There is the possibility that we have two runs that are directly
    # adjacent, ie from [s_1, e_1], [s_2, e_2] where e_1 + 1 == s_2. This
    # means we have a waiting time of 0, but it would actually be
    # included in the above list because it is of length 0. If we are
    # computing statistics of wait times, we should include these, so
    # we need to add an empty entry in.
    for i in range(1, len(properRunIntervals)):
        if properRunIntervals[i][0] == properRunIntervals[i-1][1] + 1:
            waitingTimeIntervals += [[properRunIntervals[i][0] - 1, properRunIntervals[i][0]]]
            properRunIntervals[i-1][1] += 1

    # Sort since we just added some new waiting times out of order
    if len(waitingTimeIntervals) > 0:
        order = np.argsort(np.array(waitingTimeIntervals)[:,0])
        waitingTimeIntervals = waitingTimeIntervals[order]

    return properRunIntervals, waitingTimeIntervals
        


def discretizeTrajectory(trajectoryArr, c=1, velocityThreshold=1, dt=1, minSteps=1, minDistancePerRun=0, debug=False):
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
    properRunIntervals, waitingTimeIntervals = discretizeTrajectoryIndices(trajectoryArr=trajectoryArr,
                                                                           c=c,
                                                                           velocityThreshold=velocityThreshold,
                                                                           dt=dt,
                                                                           minSteps=minSteps,
                                                                           minDistancePerRun=minDistancePerRun,
                                                                           debug=debug)

    
    if len(properRunIntervals) == 0 or len(waitingTimeIntervals) == 0:
        return np.array(trajectoryArr)[[0, -1]], np.array([len(trajectoryArr)])*dt, np.zeros(1)
        
    # Compute the start and end of the runs as the mean between the end of
    # the previous run and the start of the current run.

    # First point is just the first point in the first run interval (no
    # averaging)
    discretePoints = [trajectoryArr[properRunIntervals[0][0]]]
    # Now we take the average of the end of the previous run and the start
    # of the new one.
    for i in range(1, len(properRunIntervals)):
        discretePoints += [np.mean([trajectoryArr[properRunIntervals[i][0]],
                                    trajectoryArr[properRunIntervals[i-1][1]]], axis=0)]

    # And we add the last point of the last run
    discretePoints += [trajectoryArr[properRunIntervals[-1][1]]]

    # If we had L run intervals, we should now have L+1 discrete points
    discretePoints = np.array(discretePoints)

    waitingTimes = (waitingTimeIntervals[:,1] - waitingTimeIntervals[:,0]) * dt
    runTimes = (properRunIntervals[:,1] - properRunIntervals[:,0]) * dt
    
    return discretePoints, waitingTimes, runTimes
