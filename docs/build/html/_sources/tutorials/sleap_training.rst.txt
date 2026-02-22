SLEAP training
--------------

This tutorial covers how to train the SLEAP neural network on the data
presented in the publication. The workflow is very similar to the ones
described in the SLEAP documentation, so the value here is more describing
particular hyperparameter choices.

Data labeling
~~~~~~~~~~~~~

As with any SLEAP project, we first need to manually label (or annotate)
some frames to train the tracker on. A typical frame from our data looks
like:

.. figure:: ../images/eg_video_frame.jpg

    A sample frame from the experimental videos. Ant is in the bottom right
    quadrant.

We also need to decide on a skeleton to use in the tracking. I originally
started with just tracking the centroid of the ant, but I found that there
would be quite a lot of jitter in the final results. This is likely not only
because there is ambiguity for the actual tracker, but also because annotating
the "centroid" is difficult to do consistently. I also tried a very detailed
skeleton with each leg annotated, but given the resolution of the videos, this
did not work very well. Finally, I settled on tracking three points on the
main body of the ant: the very front of the ant ("head"), where the second set
of legs attach to the main body ("thorax"), and very back of the ant ("abdomen").

.. figure:: ../images/skeleton.png

    The skeleton used in tracking the ants.

For most of the analyses, I end up averaging these three points to get the
center of mass, but this schemes allows me to reliably annotate the positions,
and the tracking works quite well.

.. figure:: ../images/annotated_ant.png

    An example annotated ant posture in the SLEAP interface.

In total, I annotated 280 frames, using the suggested workflow of training
first on a small number of frames, and then correcting the predictions to get more
labeled data.

Network structure
~~~~~~~~~~~~~~~~~

After trying a few different network architectures, I found the best results
using a bipartite (multi-animal) model. While we only have a single ant present
in the experiments, given that the ant is so much smaller than the size of
the video frame, it is helpful to first crop to the rough location of the ant,
and then attempt to identify the body parts. The exact details of the
network can be found in the publication (SI Table 1). 

One particularly important part of the training is to enable training augmentation
to vary the angle, brightness, etc. of each training image.

Tracking results
~~~~~~~~~~~~~~~~

We get quite a small average tracking error after training, essentially indicating
that the tracker can identify the ant keypoints to the same level of accuracy
that I can manually annotate them.

.. figure:: ../images/tracking_error.png

    Average keypoint position error for the neural network tracking.
