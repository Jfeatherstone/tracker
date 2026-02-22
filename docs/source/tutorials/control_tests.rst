Control tests
-------------

This tutorial covers the analysis and conclusions drawn from the various controls
that were performed for the publication experiments.

These controls included the time of day (morning vs. afternoon) of the experiment,
the pretreatment of the ants (isolated vs. kept together), and whether the
arena was sanitized between each trial.

Notes on t-tests
~~~~~~~~~~~~~~~~

The original plan to analyze these control cases was to perform an appropriate
statistical test to make a conclusion about whether the control and test
groups differ significantly or not. The first challenge with doing this
sort of test is deciding on a quantity that feasibly will be different for
each case. 

For the cleaning comparison, we decided on using the spatial (or phase-space)
occupation maps of the experiments. These represent the exploratory behavior of the ants, how
often they revisit locations, etc. and thus seem like something that might
be able to indicate whether the ants could be affected by long-lasting pheromone
signals from previous experiments.

We thus compute these occupation maps for each trial, and then examine the
Pearson correlation coefficient, :math:`r`, between subsequent trials in
which there could be some effect of pheromones. This means the arena must
not be sanitized between the two experiments, and less than 24 hours must
elapse between the experiments (the longest lasting hindgut pheromones of
*A. gracilipes* last about 24 hours).

Time of day
~~~~~~~~~~~

Past studies have noted a difference in activity of *A. gracilipes*
foragers depending on the time of day [1], particularly that they show more
activity in the morning compared to later in the day.

.. figure:: ../images/time_of_day_test_discretized_statistics.png

Pretreatment conditions
~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/pretreatment_test_discretized_statistics.png

Pheromones (sanitizing)
~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../images/cleaning_test_discretized_statistics.png

References
~~~~~~~~~~

[1] Chong, K.-F., & Lee, C.-Y. (2009). Influences of temperature, relative
humidity and light intensity on the foraging activity of field populations
of the longlegged ant, Anoplolepis gracilipes (hymenoptera: Formicidae).
Sociobiology, 54, 531–539.

