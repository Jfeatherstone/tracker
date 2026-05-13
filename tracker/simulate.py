"""
This file includes functions to simulate run-and-tumble trajectories:

`runAndTumble()`
    Simulate a run-and-tumble particle in a 2D infinite domain.

`runAndTumbleBounded()`
    Simulate a run-and-tumble particle in a 2D finite, rectangular domain.

`runAndTumbleArbBoundary()`
    Simulate a run-and-tumble particle in an arbitrary 2D domain by specifying
    the boundary lines.

"""

import numpy as np
import numba

import random
import warnings

########################################################
#                   WAITING TIME DISTRIBUTIONS
########################################################

def gaussianDistribution(mu=0, s=.1, size=None):
    """
    Returns normally distributed values for a waiting time
    (just absolute value applied after drawing value).
    """
    return np.abs(np.random.normal(mu, s, size=size))

def exponentialDistribution(a=1, b=1, size=None):
    """
    Returns random value from an unsigned exponential distribution.
    """
    return -a*np.log(np.random.uniform(1e-10, 1, size=size)*b)

def powerLawDistribution(alpha=1, a=1, size=None):
    """
    Returns random value from an unsigned power law distribution.
    """
    # We need to define a min and max value so we just go with a very tiny
    # value for the min (pretty much allowing an arbitrarily small value
    # when considering machine error) and 1e5 as the max. This distribution
    # is only used for the waiting time distribution, so 1e5 represents a
    # reasonable maximum for that (in frames).
    return (np.random.uniform(1, 1e5, size=size)/a)**(1/(-1 - alpha))

WAITING_TIME_DISTRIBUTIONS = {"gaussian": gaussianDistribution,
                              "exponential": exponentialDistribution,
                              "power": powerLawDistribution}

def distributionFromFile(path, minValue=0, maxValue=1):
    """
    This allows us to load an arbitrary distribution from
    a file. This function returns a function that can
    be called to generate random numbers.
    """
    data = np.loadtxt(path)
    data = data / np.sum(data)
    
    valueArr = np.linspace(minValue, maxValue, len(data))
    #plt.plot(valueArr, data)
    #plt.show()

    # Unfortunately, random.choice is not supported with
    # the p argument in numba, so we can't use njit().
    def customDistribution(size=None):
        return np.random.choice(valueArr, p=data, size=size)

    return customDistribution

@numba.njit()
def isInRegion(region, position, corner=None):
    """
    Determine if a point is contained in a rectangular region, and
    if not, which boundary it violates.

    Parameters
    ----------
    region : numpy.ndarray[d]
        The size of each dimension of the region.

    position : numpy.ndarray[d]
        The position to check whether it is inside the region.

    corner : numpy.ndarray[d], optional
        The position of the corner of the region, taken to be
        the origin (`0` for every dimension) by default.

    Returns
    -------
    dimension : int
        If inside the region, a value of `-1` is returned.

        If outside the region, the index of the dimension that
        is maximally violated is returned

    above : bool
        If inside the region, a value of `False` is returned
        (arbitrary).

        If outside the region, `True` is returned if the
        boundary above the region is violated, and `False`
        is returned if the boundary below the region is violated.
    """
    if not hasattr(corner, '__iter__'):
        corner = np.zeros(len(region))

    translatedPosition = position - corner

    if not ((translatedPosition < 0).any() or (translatedPosition > region).any()):
        return -1, False

    # Which boundaries we are outside of
    belowIndices = np.where(translatedPosition < 0)[0]
    aboveIndices = np.where(translatedPosition > region)[0]

    belowDistances = np.abs(translatedPosition[belowIndices])
    aboveDistances = (translatedPosition - region)[aboveIndices]

    # If we only violate boundary above or below, we can immediately
    # return the result
    if len(belowIndices) > 0 and len(aboveIndices) == 0:
        return belowIndices[np.argmax(belowDistances)], False
    elif len(aboveIndices) > 0 and len(belowIndices) == 0:
        return aboveIndices[np.argmax(aboveDistances)], True

    # If we violate multiple boundaries, we need to see which
    # one is violated the most

    # Boolean variable for the violated boundary being above
    # or below the region
    above = np.max(aboveDistances) > np.max(belowDistances)

    # Index of the dimension where the boundary is violated
    dimensionIndex = aboveIndices[np.argmax(aboveDistances)] if above else belowIndices[np.argmax(belowDistances)]
    return dimensionIndex, above


@numba.njit()
def lineIntersection2D(a1, b1, a2, b2):
    """
    Compute the intersection of two 2-dimensional lines of the form:

    $$ a_i + b_i*t $$

    Parameters
    ----------
    a1, a2 : numpy.ndarray[d]
        Intercept points of each line.

    b1, b2 : numpy.ndarray[d]
        Direction vectors of each line.

    Returns
    -------
    intersection : numpy.ndarray[d]
        The intersection point. If the lines don't intersect,
        `[np.nan, np.nan]` will be returned.
    """
    # Compute the cross product
    # Note that when computing the cross poduct
    # in 2D, you get a *scalar* value, which represents
    # the z coordinate of the perpendicular vector
    # (since necessarily the x and y components are 0,
    # therefore no need to return a vector).
    #crossProd = np.cross(b1, b2)
    # This is not great behavior, so we actually just
    # embed the vectors in 3D and then take the z
    # component manually
    crossProd = np.cross(np.concatenate((b1, np.zeros(1))),
                         np.concatenate((b2, np.zeros(1))))[-1]

    if crossProd == 0:
       return np.repeat(np.nan, len(a1))

    # Compute the value of the parameter t, a + bt,
    # and then find the point for that t value (for both lines)
    t = np.cross(np.concatenate((a2 - a1, np.zeros(1))),
                 np.concatenate((b1, np.zeros(1))))[-1] / crossProd

    u = np.cross(np.concatenate((a2 - a1, np.zeros(1))),
                 np.concatenate((b2, np.zeros(1))))[-1] / crossProd

    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
        intersectionPoint = a1 + u*b1
        # Could also do the equivalent with the other line
        # intersectionPoint = a2 + t*b2
        return intersectionPoint
    else:
        return np.repeat(np.nan, len(a1))


def reflectPoint(p, line):
    """
    Reflect a 2D point across a line.

    For more information, see:
    https://stackoverflow.com/questions/6949722/reflection-of-a-point-over-a-line

    Parameters
    ----------
    p : array_like[2]
        The point to be reflected.
    
    line : array_like[2, 2]
        The start and end point defining the line.

    Returns
    ------- 
    pprime : numpy.ndarray[2]
        Reflected point.
    """

    a = line[0,1] - line[1,1]
    b = line[1,0] - line[0,0]
    c = - (a * line[0,0] + b * line[0,1])

    pprime = (np.array([[b**2 - a**2, -2 * a * b],
                     [-2 * a * b, a**2 - b**2]]) @ p - 2 * c * np.array([a, b])) / (a**2 + b**2)

    return pprime


def runAndTumble(totalTime, dt, v=1, x0=None,
                 tau=1, angleSigma=1, Dr=0, Dx=0,
                 waitingTimeDist='gaussian', timeKwargs={}):
    r"""
    Simulate a 2D run-and-tumble particle in an infinite domain.

    The tumbling rate is `1/tau`, meaning the probability to tumble in some
    interval `dt` is `dt / tau`. This means that the average time between
    tumbles is `tau`. This is equivalent (statistically) to having the run times be
    drawn from an exponential distribution with constant `1/tau`, or
    having the run distances be drawn from an exponential distribution
    with constant `1 / (tau * v)`.

    The tumbling angles are drawn from a von Mises distribution centered
    at zero with (inverse) width parameter `angleSigma`. A very small value
    for `angleSigma` approaches a uniform distribution, and a very large
    value gives a very peaked distribution.

    The waiting times are drawn from whatever distribution is given for
    `waitingTimeDist`, and always "during" tumbling.
    
    Uses a constant value for the velocity (`v`).

    Can include rotation and/or translational noise using the rotational
    diffusion constant (`Dr`) or the translational diffusion constant (`Dx`)
    respectively. These are the scale factors for a normal distribution 
    with unit variance.

    The equations that this function simulates are:
    
    $$ \dot x = v_0 \hat e(t) H(t) + \sqrt{2 D_x} \vec \xi(t) $$

    $$ \dot \theta = \sum_\alpha \Delta \theta_\alpha \delta (t - t_\alpha) + \sqrt{2 \pi D_r} \xi(t) H(t) $$

    $$ \hat e(t) = \sin{( \theta )} \hat x + \cos{( \theta )} \hat y $$

    $$ H(t) = \sum_\alpha \sigma(t - t_\alpha) - \sigma(t - t_\alpha - \gamma_\alpha) $$

    Parameters
    ----------
    totalTime : float
        The total simulation time.

    dt : float
        The timestep for each simulation step.
        
    v : float, optional
        The constant velocity of the walker.

    x0 : [float, float], optional
        The starting position of the walker.

        If `None`, starts at the center of the bounded space.

    tau : float
        The average time between tumbles, or the inverse of the tumbling
        rate.

    angleSigma : float
        The (inverse of the) width of the von Mises distribution from which
        angle changes are drawn from for tumbles.

    Dr : float, optional
        The rotational diffusion coefficient, adding noise the current angle.

    Dx : float, optional
        The translational diffusion coefficient, adding noise to the current
        position.

    waitingTimeDist : ['exponential', 'gaussian'] or func(size) -> numpy.ndarray[size]
        The distribution from which the waiting times are drawn from.

        Can be one of the preset distributions (see `WAITING_TIME_DISTRIBUTIONS`)
        or a custom function that takes the kwargs `size` and returns 
        random numbers with that shape.

        Parameters can be passed to this function with `timeKwargs`.

    timeKwargs : dict, optional
        The keyword arguments for the waiting time distribution function,
        `waitingTimeDist`.

    Returns
    -------
    walkArr : numpy.ndarray[N,2]
        The trajectory of the walker.
    """
    d = 2

    # Create our angle distribution
    angleFunc = lambda: random.vonmisesvariate(0, angleSigma)

    # Make sure our dt is small enough
    # If our average run time is smaller than dt, then we will probably
    # have trouble resolving runs.
    if dt / tau >= 1:
        warnings.warn('Given average run time (tau = {tau}) is smaller than the time step (dt = {dt}). Consider using a smaller time step!')

    # Validate our waiting time function
    if type(waitingTimeDist) is str:
        assert waitingTimeDist.lower() in WAITING_TIME_DISTRIBUTIONS.keys(), \
                f"Invalid preset waiting time distribution provided: {waitingTimeDist.lower()}!"

        timeFunc = WAITING_TIME_DISTRIBUTIONS[waitingTimeDist.lower()]

    elif hasattr(waitingTimeDist, '__call__'):
        timeFunc = waitingTimeDist

    else:
        raise Exception('Invalid waiting time distribution provided!')

    # Calculate the total time we will be simulating our process
    totalSteps = int(np.ceil(totalTime / dt))

    # Where we will store the current position and angle of the walker
    walkArr = np.zeros((totalSteps, 2))
    angleArr = np.zeros(totalSteps)

    # Set up initial conditions
    if hasattr(x0, '__iter__') and len(x0) == 2:
        # We have a good initial condition
        x0 = np.array(x0)

    # If we aren't given an initial condition, start
    # at the origin
    else:
        x0 = np.zeros(d)

    # This function looks quite different from the bounded ones below since
    # the angle change and position change at each timestep are all
    # generated identically (there are no, eg., boundary conditions we
    # need to check each step). As such, we can pregenerate pretty much
    # every aspect of the walk, and then just add stuff together. As a
    # result, this function is also quite a bit faster than the others.

    # Since whether we tumble at a particular time is independent of
    # any factors (probability is just dt / tau at every step) we can
    # generate the tumbling times in advance.
    isTumbling = np.random.uniform(0, 1, size=totalSteps) <= (dt / tau)

    tumblingTimes = np.where(isTumbling)[0]

    # We have to generate the waiting times now as well, since some of the
    # tumbles might be removed if they fall in the waiting time of a
    # previous tumble.
    # This array is in units of time steps, so we can directly index with
    # it.
    waitingTimes = np.array([int(timeFunc(**timeKwargs)) for _ in range(len(tumblingTimes))])

    isWaiting  = np.zeros_like(isTumbling)

    badTumbleIndices = []
    for i in range(len(tumblingTimes)):

        # Mark that we need to remove this tumble if it falls in the waiting
        # time of a previous tumble.
        if isWaiting[tumblingTimes[i]]:
            badTumbleIndices.append(i)
            continue

        # If not, set the waiting time for this tumble.
        isWaiting[tumblingTimes[i]:tumblingTimes[i]+waitingTimes[i]] = True

    # Now we remove the bad tumbles
    goodTumbleIndices = [i for i in range(len(tumblingTimes)) if i not in badTumbleIndices]
    tumblingTimes = tumblingTimes[goodTumbleIndices]

    # Now we have final versions of tumblingTimes and isWaiting.

    # Add the tumble angles
    for i in tumblingTimes:
        angleArr[i] = angleFunc()

    # Add rotation noise too
    if Dr > 0:
        angleArr += np.sqrt(2 * Dr * dt) * np.random.normal(0, 1, size=totalSteps) * (~isWaiting)
    
    # Now take the cumulative sum, giving us our final orientation for every
    # time step
    angleArr = np.cumsum(angleArr)

    # Now we can do the same thing with our step sizes
    orientationVectors = np.array([np.cos(angleArr), np.sin(angleArr)])

    # Deterministic part of steps
    steps = orientationVectors * v * dt * (~isWaiting)

    if Dx > 0:
        # Note that translational diffusion happens all the time, even
        # during waiting times.
        steps += np.sqrt(2 * Dx * dt) * np.random.normal(0, 1, size=steps.shape)

    walkArr = np.cumsum(steps.T, axis=0)

    return walkArr + x0


def runAndTumbleArbBoundary(totalTime, dt, collisionObjects, v=1, x0=None,
                           tau=1, angleSigma=1, Dr=0, Dx=0,
                           waitingTimeDist='gaussian', timeKwargs={},
                           boundaries='reflecting'):
    r"""
    Simulate a 2D run-and-tumble particle in a finite domain.

    The tumbling rate is `1/tau`, meaning the probability to tumble in some
    interval `dt` is `dt / tau`. This means that the average time between
    tumbles is `tau`. This is equivalent (statistically) to having the run times be
    drawn from an exponential distribution with constant `1/tau`, or
    having the run distances be drawn from an exponential distribution
    with constant `1/ (tau * v)`.

    The tumbling angles are drawn from a von Mises distribution centered
    at zero with (inverse) width parameter `angleSigma`. A very small value
    for `angleSigma` approaches a uniform distribution, and a very large
    value gives a very peaked distribution.

    The waiting times are drawn from whatever distribution is given for
    `waitingTimeDist`, and always "during" tumbling.
    
    Uses a constant value for the velocity (`v`).

    Can include rotation and/or translational noise using the rotational
    diffusion constant (`Dr`) or the translational diffusion constant (`Dx`)
    respectively. These are the scale factors for a uniform random
    distribution.

    The equations that this function simulates are:
    
    $$ \dot x = v_0 \hat e(t) H(t) + \sqrt{2 D_x} \vec \xi(t) + \vec F_x $$

    $$ \dot \theta = \sum_\alpha \Delta \theta_\alpha \delta (t - t_\alpha) + \sqrt{2 \pi D_r} \xi(t) H(t) + F_r $$

    $$ \hat e(t) = \sin{( \theta )} \hat x + \cos{( \theta )} \hat y $$

    $$ F_r = - a \arccos{(\hat e(t) \cdot \hat n)} $$

    $$ \vec F_x = - v_0 (\hat e(t) \cdot \hat n) \hat n $$

    $$ H(t) = \sum_\alpha \sigma(t - t_\alpha) - \sigma(t - t_\alpha - \gamma_\alpha) $$
    
    Parameters
    ----------
    totalTime : float
        The total simulation time.

    dt : float
        The timestep for each simulation step.

    region : list of numpy.ndarray[V_i,2]
        The (potentially multiple) polygons representing boundaries that the
        walker can collide with, given as lists of vertices.

        Each set of vertices should contain exactly the number of vertices
        that the shape has (ie. don't duplicate the first vertex at the end).

        The directionality of the boundary is determined by the order of
        vertices; if they are given in clockwise order, the boundary will
        reflect inside (eg. for an outer boundary), and if given in counter-
        clockwise order, will reflect outside (eg. for an internal obstacle).
        
    v : float, optional
        The constant velocity of the walker.

    x0 : [float, float], optional
        The starting position of the walker.

        If `None`, starts at the center of the bounded space.

    tau : float
        The average time between tumbles, or the inverse of the tumbling
        rate.

    angleSigma : float
        The (inverse of the) width of the von Mises distribution from which
        angle changes are drawn from for tumbles.

    Dr : float, optional
        The rotational diffusion coefficient, adding noise the current angle.

    Dx : float, optional
        The translational diffusion coefficient, adding noise to the current
        position.

    waitingTimeDist : ['exponential', 'gaussian'] or func(size) -> numpy.ndarray[size]
        The distribution from which the waiting times are drawn from.

        Can be one of the preset distributions (see `WAITING_TIME_DISTRIBUTIONS`)
        or a custom function that takes the kwargs `size` and returns 
        random numbers with that shape.

        Parameters can be passed to this function with `timeKwargs`.

    timeKwargs : dict, optional
        The keyword arguments for the waiting time distribution function,
        `waitingTimeDist`.

    boundaries : {'reflecting', 'aligning'}
        The type of boundary conditions to use for the finite domain.

        `'reflecting'` means the particle is reflected (elastically) from
        the boundary.

        `'aligning'` means the particle will align itself with the
        boundary upon collision.

    Returns
    -------
    walkArr : numpy.ndarray[N,2]
        The trajectory of the walker.
    """
    d = 2

    # Create our angle distribution
    angleFunc = lambda: random.vonmisesvariate(0, angleSigma)

    # Make sure our dt is small enough
    # If our average run time is smaller than dt, then we will probably
    # have trouble resolving runs.
    if dt / tau >= 1:
        warnings.warn('Given average run time (tau = {tau}) is smaller than the time step (dt = {dt}). Consider using a smaller time step!')

    # Validate our boundary conditions
    assert boundaries.lower() in ['reflecting', 'aligning'], f'Invalid boundary condition provided: {boundaries}'

    # Validate our waiting time function
    if type(waitingTimeDist) is str:
        assert waitingTimeDist.lower() in WAITING_TIME_DISTRIBUTIONS.keys(), \
                f"Invalid preset waiting time distribution provided: {waitingTimeDist.lower()}!"

        timeFunc = WAITING_TIME_DISTRIBUTIONS[waitingTimeDist.lower()]

    elif hasattr(waitingTimeDist, '__call__'):
        timeFunc = waitingTimeDist

    else:
        raise Exception('Invalid waiting time distribution provided!')

    # Compute the lines that represent the boundaries of the
    # region. Each element in the list is a separate object.
    boundaryLines = []
    boundaryNormals = []
    for i in range(len(collisionObjects)):

        points = collisionObjects[i]
        # We want to duplicate the first point at the end so that we can
        # easily compute the lines. You could probably also use np.roll
        points = np.concatenate((points, points[:1]), axis=0)

        lines = np.array([points[1:], points[:-1]])
        lines = np.swapaxes(lines, 0, 1)[:,::-1]

        normals = np.cross(points[1:] - points[:-1], (0, 0, 1))[:,:2]
        # Normalize
        normals = (normals.T / np.sqrt(np.sum(normals**2, axis=-1))).T

        boundaryLines += list(lines)
        boundaryNormals += list(normals)

    # We access these as, eg. boundaryLines[i], which will give the two
    # points that form the line for the ith boundary.
    # Similarly, boundaryNormals[i] is a unit vector (in 2D) representing
    # the normal to the boundary.
    boundaryLines = np.array(boundaryLines)
    boundaryNormals = np.array(boundaryNormals)

    # We want to compute the size of the region to be able to give
    # appropriate padding when dealing with the rigid body force at the
    # boundary. Typically we use 1/1000 of this region size as the padding.
    bottomLeftCorner = np.array((np.min(boundaryLines[:,:,0].flatten()), np.min(boundaryLines[:,:,1].flatten())))
    topRightCorner = np.array((np.max(boundaryLines[:,:,0].flatten()), np.max(boundaryLines[:,:,1].flatten())))
    regionSize = np.linalg.norm(topRightCorner - bottomLeftCorner)

    # Calculate the total time we will be simulating our process
    totalSteps = int(np.ceil(totalTime / dt))

    # Where we will store the current position and angle of the walker
    walkArr = np.zeros((totalSteps, 2))
    angleArr = np.zeros(totalSteps)

    # Set up initial conditions
    if hasattr(x0, '__iter__') and len(x0) == 2:
        walkArr[0] = x0

    # If we aren't given an initial condition, start
    # in the center of the region.
    else:
        walkArr[0] = (topRightCorner + bottomLeftCorner) / 2

    # Start with a totally random angle.
    angleArr[0] = np.random.uniform(-np.pi, np.pi)
    # Start with zero waiting time.
    currentWaitingTime = 0
    
    for i in range(1, totalSteps):

        # Reduce the waiting time (if we are waiting; otherwise, this is already
        # negative and it doesn't affect anything).
        currentWaitingTime -= dt

        # Since translational diffusion happens even during waiting times,
        # we have to add that now.
        dxDiff = np.sqrt(2 * Dx * dt) * np.random.normal(0, 1, size=2)

        if currentWaitingTime > 0:
            walkArr[i] = walkArr[i-1] + dxDiff
            angleArr[i] = angleArr[i-1]
            continue

        # Compute the deterministic step
        # dx = v * e(t)
        # Angle is defined from the x axis (ie. orientation of 0 leads to
        # orientation vector (1,0)).
        e = np.array([np.cos(angleArr[i-1]), np.sin(angleArr[i-1])])
        # Only add the deterministic part (first term) if we aren't in a
        # waiting time, but add the spatial noise (Dx, second term)
        # regardless.
        dx = v * e * dt + dxDiff
        walkArr[i] = walkArr[i-1] + dx
        
        # Most of the time we just copy over the previous angle, potentially
        # with some noise set by the rotational diffusion coefficient.
        angleArr[i] = angleArr[i-1] + np.sqrt(2 * Dr * dt) * np.random.normal(0, 1)
        
        # Check if we should tumble
        if np.random.uniform(0, 1) <= dt / tau:
            currentWaitingTime = timeFunc(**timeKwargs) * dt
            angleArr[i] += angleFunc()

        # Check if the step crosses any boundaries. Note that this only
        # assumes a single boundary could be violated at once. This is
        # usually a good assumption assuming you have a small enough timestep.
        violatedBoundary = -1
        for j in range(len(boundaryLines)):
            intersectionPoint = lineIntersection2D(boundaryLines[j,0], boundaryLines[j,1] - boundaryLines[j,0],
                                                   walkArr[i-1], dx)

            if np.isnan(intersectionPoint).any():
                continue

            violatedBoundary = j
            break

        # If no boundaries are violated, we are done.
        if violatedBoundary == -1:
            continue

        # If we have crossed a boundary, we set the current position to
        # the intersection point (with some small padding)
        walkArr[i] = intersectionPoint + boundaryNormals[violatedBoundary]*1e-3*regionSize

        # Now adjust the angle so we are facing along the wall, or are
        # reflected by it, depending on the boundary condition.
        # Find the angle between the current orientation and the wall.
        # Recalculate e since we updated the angle earlier.
        e_hat = np.array([np.cos(angleArr[i]), np.sin(angleArr[i])])
        n_hat = boundaryNormals[violatedBoundary]

        wallAngle = np.sign(np.cross(n_hat, e_hat)) * np.arcsin(np.dot(n_hat, e_hat))

        if boundaries == 'reflecting':
            # For reflective boundary conditions, we add to the angle
            # double of the angle between the orientation vector
            # and the wall, so we are reflected.
            angleArr[i] += wallAngle * 2 

        elif boundaries == 'aligning':
            # For aligning boundary conditions, we add to the angle
            # exactly the angle between the orientation vector and the wall
            angleArr[i] += wallAngle

    return walkArr


def runAndTumbleBounded(totalTime, dt, region, v=1, x0=None,
                        tau=1, angleSigma=1, Dr=0, Dx=0,
                        waitingTimeDist='gaussian', timeKwargs={},
                        boundaries='reflecting'):
    r"""
    Simulate a 2D run-and-tumble particle in a finite, rectangular domain.

    The tumbling rate is `1/tau`, meaning the probability to tumble in some
    interval `dt` is `dt / tau`. This means that the average time between
    tumbles is `tau`. This is equivalent (statistically) to having the run times be
    drawn from an exponential distribution with constant `1/tau`, or
    having the run distances be drawn from an exponential distribution
    with constant `1/ (tau * v)`.

    The tumbling angles are drawn from a von Mises distribution centered
    at zero with (inverse) width parameter `angleSigma`. A very small value
    for `angleSigma` approaches a uniform distribution, and a very large
    value gives a very peaked distribution.

    The waiting times are drawn from whatever distribution is given for
    `waitingTimeDist`, and always "during" tumbling.
    
    Uses a constant value for the velocity (`v`).

    Can include rotation and/or translational noise using the rotational
    diffusion constant (`Dr`) or the translational diffusion constant (`Dx`)
    respectively. These are the scale factors for a uniform random
    distribution.

    The equations that this function simulates are:
    
    $$ \dot x = v_0 \hat e(t) H(t) + \sqrt{2 D_x} \vec \xi(t) + \vec F_x $$

    $$ \dot \theta = \sum_\alpha \Delta \theta_\alpha \delta (t - t_\alpha) + \sqrt{2 \pi D_r} \xi(t) H(t) + F_r $$

    $$ \hat e(t) = \sin{( \theta )} \hat x + \cos{( \theta )} \hat y $$

    $$ F_r = - a \arccos{(\hat e(t) \cdot \hat n)} $$

    $$ \vec F_x = - v_0 (\hat e(t) \cdot \hat n) \hat n $$

    $$ H(t) = \sum_\alpha \sigma(t - t_\alpha) - \sigma(t - t_\alpha - \gamma_\alpha) $$
    
    This function is actually just a wrapper around `runAndTumbleArbBoundary`,
    which allows for simulation with arbitrary boundaries.

    Parameters
    ----------
    totalTime : float
        The total simulation time.

    dt : float
        The timestep for each simulation step.

    region : [float, float]
        The extent of the finite region in each dimension, ie. the
        corner of the region opposite from [0,0].
        
    v : float, optional
        The constant velocity of the walker.

    x0 : [float, float], optional
        The starting position of the walker.

        If `None`, starts at the center of the bounded space.

    tau : float
        The average time between tumbles, or the inverse of the tumbling
        rate.

    angleSigma : float
        The (inverse of the) width of the von Mises distribution from which
        angle changes are drawn from for tumbles.

    Dr : float, optional
        The rotational diffusion coefficient, adding noise the current angle.

    Dx : float, optional
        The translational diffusion coefficient, adding noise to the current
        position.

    waitingTimeDist : ['exponential', 'gaussian'] or func(size) -> numpy.ndarray[size]
        The distribution from which the waiting times are drawn from.

        Can be one of the preset distributions (see `WAITING_TIME_DISTRIBUTIONS`)
        or a custom function that takes the kwargs `size` and returns 
        random numbers with that shape.

        Parameters can be passed to this function with `timeKwargs`.

    timeKwargs : dict, optional
        The keyword arguments for the waiting time distribution function,
        `waitingTimeDist`.

    boundaries : {'reflecting', 'aligning'}
        The type of boundary conditions to use for the finite domain.

        `'reflecting'` means the particle is reflected (elatically) from
        the boundary.

        `'aligning'` means the particle will align itself with the
        boundary upon collision.

    Returns
    -------
    walkArr : numpy.ndarray[N,2]
        The trajectory of the walker.
    """
    # Note that this must be in clockwise order, otherwise the boundary
    # will only take effect from the opposite side.
    boundaryPoints = np.array([[0, 0],
                               [0, region[1]],
                               [region[0], region[1]],
                               [region[0], 0]])

    return runAndTumbleArbBoundary(totalTime, dt, [boundaryPoints], v, x0,
                                   tau, angleSigma, Dr, Dx,
                                   waitingTimeDist, timeKwargs,
                                   boundaries)
