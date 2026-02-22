SLEAP tracking and postprocessing
---------------------------------

This tutorial covers the tracking and post-processing workflow for the
data in the publication. Unless you are working with your own data, you
probably won't need to follow any of the steps here, as the final post-processed
data is available to download (to be used in the rest of the tutorials).

Excepting the tracking done with SLEAP, all of the code that implements
these steps can be found in the notebook ``00_PreprocessData.ipynb``.

SLEAP tracking
~~~~~~~~~~~~~~

Tracking the videos with SLEAP is relatively easy, though since there is
quite a large amount (~100), I created a script to do this on my institute's
cluster.

.. collapse:: Click here for the script.

    This script was written for the ``slurm`` job scheduler, though it
    can also just be run as a bash script. Ideally it should be run
    on a computer with a GPU.

    .. code-block:: bash

        #!/bin/bash

        #SBATCH -p gpu
        #SBATCH -t 72:00:00
        #SBATCH --mem=64G
        #SBATCH --gres=gpu:1
        #SBATCH --job-name=SLEAP-Track

        ##############################################
        # Script to run a sleap training job
        #
        # The first argument should be a partial match to the
        # model file name. We use a partial match so that if
        # we have a two part model (centroid and centered instance)
        # we can match both with the same partial string.
        #
        # The following argument(s) should be the path to the video(s).
        # 
        # eg. using slurm:
        # sbatch track.slurm model_name_partial /path/to/video1 /path/to/video2 ...
        #
        # eg. as a bash script:
        # ./track.slurm model_name_partial /path/to/video1 /path/to/video2 ...
        #
        # Note that this will likely not work if you try
        # to resume training from a previous checkpoint,
        # since it won't be able to find the previous
        # model files.
        ##############################################

        ##############################################
        # Parameters
        ##############################################

        # Will copy the trajectories over to the following location after finishing
        # Can be a remote location, eg: "external:/path/to/output/"
        outputDir="/path/to/output/"

        # The directory where sleap should search for models
        modelsDir="/path/to/models/"

        # Somewhere to create a temporary folder. If you have a work partition
        # on a cluster or something, this should point there. ie. some place that
        # you have read/write permissions.
        tempLocation="/path/to/temp/"

        # The name of your conda environment that has sleap
        sleapEnv="sleap"

        # If you need to source a file (eg. .bashrc, .zshrc) in order to
        # load conda, you can do that here.
        source /path/to/your/source_file  > /dev/null 2>&1

        ##############################################
        # End parameters
        ##############################################

        # First, search for the full model path
        # We pass as an argument the partial name of the model,
        # since if we are doing multi-animal, we might have
        # more than one model.
        modelFiles=`find $modelsDir -name "*$1*" -print`

        modelArgs=""
        for modelName in $modelFiles;
        do
            echo "Using model: $modelName"
            modelArgs+="-m $modelName "
        done

        # Make sure we found something
        # Check that the length of the string is zero
        if [ ${#modelArgs} -eq 0 ]; then
            echo "Invalid model name provided! No models found in $modelsDir that match partial string \"$1\""
            exit 1
        fi

        # Make sure the video file(s) exist
        # Remove the first argument, since that was the model name
        shift
        for video in "$@"; do
            if [ ! -f $video ]; then
                echo "Video does not exist: $video"
                exit 1
            fi
        done

        echo ""
        echo "Running tracking on videos:"
        for video in "$@"; do
            echo "       $video"
        done


        # Create a temporary directory for this job and save the name
        tempdir=$(mktemp -d "${tempLocation}/temp.XXXXXX")

        echo ""
        echo "Created temporary directory: $tempdir"

        # Enter the temporary directory
        cd $tempdir

        # Activate your sleap conda environment
        conda activate $sleapEnv

        # Now loop over each video and perform the tracking
        for videoPath in "$@"; do
            echo ""
            echo "Tracking video: $videoPath"
            # Remove the path
            outputPath="$(basename $videoPath)"
            # Remove the extension
            outputPath="${outputPath%.*}_tracked"
            echo "Output file: $outputPath"

            sleap-track $videoPath -o $outputPath $modelArgs \
                --video.dataset video --video.input_format channels_last \
                --tracking.tracker simple --tracking.similarity centroid

            # Note that sleap will automatically add a ".slp" to the
            # output name, so we need to add that here.
            trueOutputPath="${outputPath}.slp"

            # Copy our result back to the output. We use "scp" to copy the data
            # back as we may have a remote server.
            if [ -f $trueOutputPath ]; then
                echo "Copying file: $trueOutputPath"
                scp $trueOutputPath $outputDir/
            fi
        done

        # Clean up by removing our temporary directory
        echo "Cleaning up temporary directory"
        rm -r $tempdir

        echo ""
        echo "Done!"

    .. note::

        I made several changes to this script to make it more general when
        copying it here, so if it doesn't work exactly as written, sorry...
        But you should only have to make some small corrections (hopefully).

After running the tracking, you should have a ``.slp`` file for each video
tracked, which includes the tracks detected in each video. Note that the
script above only using the centroid tracker (tracker in the sense of identifying
posture detections through time as the same ant). This means that for each
video, we actually have a very large amount of "individual" ants.

.. figure:: ../images/initial_tracks.png

    The number of identified ants in a single video. These are of course all
    the same ant.

So one of the first things we need to do is to clean up these tracks, and give
a consistent identity across the whole video. This isn't as easy as just 
assigning each position to the same identity, since there sometimes are spurious
identifications (see the horizontal lines in the image above). Maybe there is
a way to just better calibrate SLEAP's tracker, but I just ended up using
a custom one, which factors in the distance from previous positions, as well
as the score of the detection to connect the trajectories.

For more information, see :meth:`tracker.trackSleapFile`.

Identifying segments
~~~~~~~~~~~~~~~~~~~~

While the tracker helps to make the posture instances more continuous through
time, there are inevitably going to be gaps in the trajectories. These can be
caused by small tracking errors or failures, but also by the ant leaving the
field of view of the camera (possible because the ants can climb the walls).
To deal with these gaps in a consistent way, we set a maximum limit to how
much we (linearly) interpolate missing data points. I typically use 30 frames
(0.5s for 60Hz) throughout the analysis unless specified otherwise. When we
have a gap larger than this amount, we separate the trajectory before and
after into different "segments", meaning a single experiment can have
potentially many trajectories. We generally see that, for each experiment,
there will be a few longer segments, and then many shorter ones comprising
short bouts of motion.

.. subfigure::

    .. image:: ../images/segment_lengths.png

    .. image:: ../images/jump_sizes.png

    Histograms of the duration of each segment for ant trajectories (left)
    and the largest interpolated jump in each segment (right).

Next, we observe that there are a few segments that give tracking data outside
of the enclosure; this could be due to spurious detections, or could be 
from tracker the ant as it enters the enclosure. Either way, we want to
remove them, so we remove points beyond the manually-defined bounds of
the arena.

.. figure:: ../images/overlay.png

    All trajectories overlaid; some show detections outside of the arena.

Attaching metadata and cleaning up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point, we have all of our segments identified, so we just need to
package the data in a way that is easy to analyze later. I decided to use
the `hdf5 format <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_,
since it allows the metadata to be kept alongside each segment. This includes
information about the experimental conditions, the SLEAP tracking, and the
post-processing.
