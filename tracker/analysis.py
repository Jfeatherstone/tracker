import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from .discretize import discretizeTrajectoryIndices

def autocorrelation(x, minLag=0, maxLag=100, dt=1, dtau=1, normalize=True):
    r"""
    Compute the autocorrelation of some vector.

    :math:`\langle x(t) x(t + \tau) \rangle`

    Parameters
    ----------
    x : numpy.ndarray[N]
        The vector to compute the autocorrelation for.

    minLag : float
        The minimum lag to compute, in the same units as `dt`.

    maxLag : float
        The maximum lag to compute, in the same units as `dt`.

    dt : float
        The time difference between each point in `x`.

    dtau : float
        The interval at which to sample the autocorrelation, in the same
        units as `dt`.

    normalize : bool
        Whether to normalize the autocorrelation values by the first
        entry.

    Returns
    -------
    lagArr : numpy.ndarray[M]
        The time lags for which the autocorrelation is computed.

    autocorr : numpy.ndarray[M]
        The autocorrelation at each time lag.

    """

    # These are indices
    lagArr = np.arange(minLag / dt, maxLag / dt, dtau / dt).astype(np.int64)
    lagArr = np.unique(lagArr)

    # Make sure we don't try to compute lags longer than the length of
    # the vector
    lagArr = lagArr[lagArr < len(x)-1]

    autocorrArr = np.zeros(len(lagArr))

    for i in range(len(lagArr)):
        autocorrArr[i] = np.mean(x[:len(x)-lagArr[i]] * x[lagArr[i]:])

    if normalize:
        autocorrArr = autocorrArr / autocorrArr[0]

    return lagArr * dt, autocorrArr


def computeAngles(trajectory, dt=1, minVelocityThreshold=0):
    r"""
    Compute the angle difference (turn angles) throughout a 2D trajectory.

    This does not necessarily maintain the same indexing as the input trajectory,
    since it is primarily for compute distributions and statistics.

    Parameters
    ----------
    trajectory : numpy.ndarray[N,2]
        The trajectory to compute the angles for.

    dt : float, optional
        The time different between each point in the trajectory. Used to compute
        the velocity and threshold based on ``minVelocityThreshold``.

    minVelocityThreshold : float, optional
        The velocity threshold below which an angle won't be included in
        the final array of angles. Useful since you may see huge angle changes
        which are actually just jitter if the trajectory remains (nearly) stationary
        for some period of time.

    Returns
    -------
    angles : numpy.ndarray[M]
        All of the (valid) turn angles throughout the trajectory.

    """
    # Compute the velocity for the trajectory
    velocityArr = (trajectory[1:] - trajectory[:-1]) / dt
    velocityMagnitudes = np.sqrt(np.sum(velocityArr**2, axis=-1))

    velocityArr = velocityArr[velocityMagnitudes > minVelocityThreshold]
    velocityMagnitudes = velocityMagnitudes[velocityMagnitudes > minVelocityThreshold]

    # Make unit vectors
    velocityArr[:,0] = velocityArr[:,0] / velocityMagnitudes
    velocityArr[:,1] = velocityArr[:,1] / velocityMagnitudes

    # Now take angle differences for the steps
    # The clip is to avoid numerical errors
    angleDifferenceArr = np.array([np.arccos(np.clip(np.dot(velocityArr[i], velocityArr[i+1]), -1., 1.)) for i in range(len(velocityArr)-1)])

    # To compute the sign of the angle, we need
    # to use the cross product
    signArr = np.array([np.sign(np.cross(velocityArr[i], velocityArr[i+1])) for i in range(len(velocityArr)-1)])
    angleDifferenceArr = angleDifferenceArr * signArr

    return angleDifferenceArr[~np.isnan(angleDifferenceArr)]


def exponential(x, x0, A):
    """
    Exponential function used in fitting the rotational diffusion and/or
    persistence time.
    """
    return np.exp(-x / x0) * A


def computeRotationalDiffusion(trajectory, c, v, minStepsPerRun, minDistancePerRun, t=None, minLag=0, maxLag=2, dtau=0.1):
    """
    Compute the rotational diffusion coefficient, assuming
    that the trajectory data follows a run-and-tumble scheme.

    This is done by discretizing the trajectory as a run-and-tumble,
    then computing the autocorrelation of just the runs. Since each
    individual run might not have enough data to fit by itself, the
    data from all runs are pooled together.

    Parameters
    ----------
    trajectory : numpy.ndarray[N,2] or list of numpy.ndarray[N,2]
        The trajectory to compute the rotational diffusion coefficient
        for.

        Can also be a list of disjoint segments representing the same
        trajectory (for example, if there are gaps in a trajectory that
        should be avoided in calculate the persistence, but you want
        to average over the entire collection of segments).

    theta : float
        The angle threshold parameter used for discretizing the
        trajectory, in radians.

    v : float
        The velocity threshold parameter used for discretizing the
        trajectory.

    minStepsPerRun : int
        The minimum number of steps per run for discretizing the
        trajectory.

    minDistancePerRun : float
        The minimum distance per run for discretizing the
        trajectory. Given in whatever units ``trajectory`` is given in.

    t : float or numpy.ndarray[N,2] or list of numpy.ndarray[N,2], optional
        The time points at which the data is sampled. If all samples are
        evenly spaced, can be a single float representing the time difference.

        If ``trajectory`` is given as a list of segments, this should be structured
        similarly.

        If not given, all samples will be assumed to be spaced evenly, and ``v`` will
        be given in units of distance/frames (where the distance unit is whatever
        the units of ``trajectory`` are.

    minLag : float
        The minimum time lag to compute the autocorrelation for, given in the
        same units as ``t``.

    maxLag : float
        The max time lag to compute the autocorrelation for, given in the same units as ``t``.

    dtau : float
        The spacing between sampled time lags to compute the rotational diffusion
        using, given in the same units as ``t``.

    Returns
    -------
    Dr : float or numpy.nan
        The rotational diffusion coefficient computed by pooling the autocorrelation
        data for all of the trajectories. If fitting failed or some other issue
        arose, ``np.nan`` will be returned.

    """
    if type(trajectory) is list:
        segmentList = trajectory

    else:
        segmentList = [trajectory]

    if not hasattr(t, '__iter__'):
        # If t is None, we just use a spacing of 1
        if not t:
            dt = 1

        # Otherwise we expect that t is the dt value
        else:
            dt = t

        # Now we just generate an array of times the same shape as trajectory
        timeList = [np.arange(len(segmentList[i]))*dt for i in range(len(segmentList))]

    else:
        # If we are given a proper t array, we should be able to copy it directly
        if type(t) is list and type(trajectory) is list:
            # Make sure it has the same shape
            for i in range(len(trajectory)):
                assert len(t[i]) == len(trajectory[i])

            timeList = t

        else:
            # Otherwise, we only have a single segment, so just make sure they are
            # the same length, and then save.
            assert len(t) == len(segmentList[0])
            timeList = [t]

    # Technically this could change between segments, but it really shouldn't
    dt = np.min(timeList[0][1:] - timeList[0][:-1])

    # Setup our lag arrays and the bins we will be adding data to throughout
    # the process.
    lagArr = np.arange(minLag / dt, maxLag / dt, dtau / dt).astype(np.int64)
    lagArr = np.unique(lagArr)

    autocorrSamplesArr = [[] for _ in range(len(lagArr))]

    # Now we actually start working with the trajectories
    for i in range(len(segmentList)):

        # First, we need to discretize the segment
        data = segmentList[i]
        time = timeList[i]

        runIntervals, waitIntervals = discretizeTrajectoryIndices(data,
                                                                  c=c,
                                                                  velocityThreshold=v,
                                                                  dt=dt,
                                                                  minSteps=minStepsPerRun,
                                                                  minDistancePerRun=minDistancePerRun,
                                                                  debug=False)

        # If we have detected no runs, we continue on to the next segment
        if len(runIntervals) == 0:
            continue

        for j in range(1, len(runIntervals)):
            runData = data[runIntervals[j][0]:runIntervals[j][1]]

            angles = computeAngles(runData, dt, v)
            cumAngles = np.cumsum(angles)
            cosArr = np.cos(cumAngles)

            # Compute the autocorrelation
            for k in range(len(lagArr)):
                # For shorter runs, we can only compute some of the smaller bins
                if lagArr[k] > len(cosArr) - 1:
                    break

                autocorrSamplesArr[k] += list(cosArr[:len(cosArr) - lagArr[k]] * cosArr[lagArr[k]:])

    # Compute the mean of the autocorrelation to finish it
    autocorrArr = np.array([np.nanmean(autocorr) if len(autocorr) > 0 else np.nan for autocorr in autocorrSamplesArr])

    autocorrArr = autocorrArr / autocorrArr[0]

    tArr = lagArr * dt
    # Fit with an exponential
    try:
        pOpt, pCov, info, mesg, success = curve_fit(exponential, tArr, autocorrArr,
                                                    full_output=True, p0=(1, 1))
        # 1/param since diffusion coefficient has units of inverse seconds
        Dr = 1/pOpt[0]
        pErr = np.sqrt(np.diag(pCov))

        # DEBUG
        #plt.plot(tArr, autocorrArr)
        #plt.plot(tArr, exponential(tArr, *pOpt))
        #plt.yscale('log')
        #plt.show()

        if Dr <= 0:
            return np.nan

        return Dr

    except:
        return np.nan


def computePersistence(trajectory, v, t=None, minLag=0, maxLag=2, dtau=0.1):
    """
    Compute the rotational diffusion coefficient, assuming
    that the trajectory data follows a run-and-tumble scheme.

    This is done by discretizing the trajectory as a run-and-tumble,
    then computing the autocorrelation of just the runs. Since each
    individual run might not have enough data to fit by itself, the
    data from all runs are pooled together.

    Parameters
    ----------
    trajectory : numpy.ndarray[N,2] or list of numpy.ndarray[N,2]
        The trajectory to compute the rotational diffusion coefficient
        for.

        Can also be a list of disjoint segments representing the same
        trajectory (for example, if there are gaps in a trajectory that
        should be avoided in calculate the persistence, but you want
        to average over the entire collection of segments).

    t : float or numpy.ndarray[N,2] or list of numpy.ndarray[N,2], optional
        The time points at which the data is sampled. If all samples are
        evenly spaced, can be a single float representing the time difference.

        If ``trajectory`` is given as a list of segments, this should be structured
        similarly.

        If not given, all samples will be assumed to be spaced evenly, and ``v`` will
        be given in units of distance/frames (where the distance unit is whatever
        the units of ``trajectory`` are.

    minLag : float
        The minimum time lag to compute the autocorrelation for, given in the
        same units as ``t``.

    maxLag : float
        The max time lag to compute the autocorrelation for, given in the same units as ``t``.

    dtau : float
        The spacing between sampled time lags to compute the rotational diffusion
        using, given in the same units as ``t``.

    Returns
    -------
    persistence : float or numpy.nan
        The persistence time computed by pooling the autocorrelation data for
        all of the trajectories. If fitting failed or some other issue arose,
        ``np.nan`` will be returned.

    """
    if type(trajectory) is list:
        segmentList = trajectory

    else:
        segmentList = [trajectory]

    if not hasattr(t, '__iter__'):
        # If t is None, we just use a spacing of 1
        if not t:
            dt = 1

        # Otherwise we expect that t is the dt value
        else:
            dt = t

        # Now we just generate an array of times the same shape as trajectory
        timeList = [np.arange(len(segmentList[i]))*dt for i in range(len(segmentList))]

    else:
        # If we are given a proper t array, we should be able to copy it directly
        if type(t) is list and type(trajectory) is list:
            # Make sure it has the same shape
            for i in range(len(trajectory)):
                assert len(t[i]) == len(trajectory[i])

            timeList = t

        else:
            # Otherwise, we only have a single segment, so just make sure they are
            # the same length, and then save.
            assert len(t) == len(segmentList[0])
            timeList = [t]

    # Technically this could change between segments, but it really shouldn't
    dt = np.min(timeList[0][1:] - timeList[0][:-1])

    # Setup our lag arrays and the bins we will be adding data to throughout
    # the process.
    lagArr = np.arange(minLag / dt, maxLag / dt, dtau / dt).astype(np.int64)
    lagArr = np.unique(lagArr)

    autocorrSamplesArr = [[] for _ in range(len(lagArr))]

    # Now we actually start working with the trajectories
    for i in range(len(segmentList)):

        # First, we need to discretize the segment
        data = segmentList[i]
        time = timeList[i]

        angles = computeAngles(data, dt, v)
        cumAngles = np.cumsum(angles)
        cosArr = np.cos(cumAngles)

        # Compute the autocorrelation
        for k in range(len(lagArr)):
            # For shorter runs, we can only compute some of the smaller bins
            if lagArr[k] > len(cosArr) - 1:
                break

            autocorrSamplesArr[k] += list(cosArr[:len(cosArr) - lagArr[k]] * cosArr[lagArr[k]:])

    # Compute the mean of the autocorrelation to finish it
    autocorrArr = np.array([np.nanmean(autocorr) if len(autocorr) > 0 else np.nan for autocorr in autocorrSamplesArr])

    autocorrArr = autocorrArr / autocorrArr[0]

    tArr = lagArr * dt
    # Fit with an exponential
    try:
        pOpt, pCov, info, mesg, success = curve_fit(exponential, tArr, autocorrArr,
                                                    full_output=True, p0=(1, 1))

        persistence = pOpt[0]
        pErr = np.sqrt(np.diag(pCov))

        # DEBUG
        #plt.plot(tArr, autocorrArr)
        #plt.plot(tArr, exponential(tArr, *pOpt))
        #plt.yscale('log')
        #plt.show()

        if persistence <= 0:
            return np.nan

        return persistence

    except:
        return np.nan


def computeMSD(trajectory, t=None, minLag=0, maxLag=30, nbins=30, logbins=True):
    """
    Compute the mean-squared displacement of a trajectory as a function of
    the time lag between samples.

    Slightly more complicated than one might expect, since we might have
    time jumps in the sampling, and we want to be able to count those properly.

    Parameters
    ----------
    trajectory : numpy.ndarray[N,2] or list of numpy.ndarray[N,2]
        The trajectory to compute the rotational diffusion coefficient
        for.

        Can also be a list of disjoint segments representing the same
        trajectory (for example, if there are gaps in a trajectory that
        should be avoided in calculate the persistence, but you want
        to average over the entire collection of segments).

    t : float or numpy.ndarray[N,2] or list of numpy.ndarray[N,2], optional
        The time points at which the data is sampled. If all samples are
        evenly spaced, can be a single float representing the time difference.

        If ``trajectory`` is given as a list of segments, this should be structured
        similarly.
 
        If not given, all samples will be assumed to be spaced evenly.

    minLag : float
        The minimum time lag to compute the MSD for, given in the
        same units as ``t``.

    maxLag : float
        The max time lag to compute the MSD for, given in the same units as ``t``.

    nbins : int
        The number of time lags to sample in the specified range.

    logbins : bool
        Whether to space the time lags linearly (False) or logarithmically
        (True).

    Returns
    -------
    tArr : numpy.ndarray[nbins]
        The time lags the MSD is computed for

    msdArr : numpy.ndarray[nbins]
        The MSD for each time lag, averaged across all trajectories provided.

    """
    if type(trajectory) is list:
        segmentList = trajectory

    else:
        segmentList = [trajectory]

    if not hasattr(t, '__iter__'):
        # If t is None, we just use a spacing of 1
        if not t:
            dt = 1

        # Otherwise we expect that t is the dt value
        else:
            dt = t

        # Now we just generate an array of times the same shape as trajectory
        timeList = [np.arange(len(segmentList[i]))*dt for i in range(len(segmentList))]

    else:
        # If we are given a proper t array, we should be able to copy it directly
        if type(t) is list and type(trajectory) is list:
            # Make sure it has the same shape
            for i in range(len(trajectory)):
                assert len(t[i]) == len(trajectory[i])

            timeList = t

        else:
            # Otherwise, we only have a single segment, so just make sure they are
            # the same length, and then save.
            assert len(t) == len(segmentList[0])
            timeList = [t]

    # Technically this could change between segments, but it really shouldn't
    dt = np.min(timeList[0][1:] - timeList[0][:-1])

    # Setup our lag arrays and the bins we will be adding data to throughout
    # the process.
    if not logbins:
        lagArr = (np.linspace(minLag, maxLag, nbins)/dt).astype(np.int64)
    else:
        lagArr = (np.logspace(np.log10(minLag/dt), np.log10(maxLag/dt), nbins)).astype(np.int64)

    #lagArr = np.unique(lagArr)

    msdSamplesArr = [[] for _ in range(len(lagArr))]

    # Now we actually start working with the trajectories
    for i in range(len(segmentList)):

        # First, we need to discretize the segment
        data = segmentList[i]
        time = timeList[i]

        # Compute the autocorrelation
        for k in range(len(lagArr)):
            lag = lagArr[k]

            # For shorter runs, we can only compute some of the smaller bins
            if lag > len(data) - 1:
                break
 
            timeDiffArr = time[lag:] - time[:len(time)-lag]
            timeDiffIndices = np.around(timeDiffArr / dt).astype(np.int64)
            goodIndices = np.where(timeDiffIndices == lag)[0]

            if (len(goodIndices) == 0) or (lag == 0):
                continue

            msdSamplesArr[k] += list(np.sum((data[lag:] - data[:-lag])**2, axis=-1)[goodIndices])

    # Compute the mean of the autocorrelation to finish it
    msdArr = np.array([np.nanmean(msd) if len(msd) > 0 else np.nan for msd in msdSamplesArr])

    tArr = lagArr * dt

    return tArr, msdArr
