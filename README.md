Ant tracker
===

This repository includes the code to load and analyze laboratory ant tracking
experiments.

Usage
---

The dataset can be retrieved from Zenodo. It will include a copy of the
code as well, though this repository will likely be updated more regularly.

Clone this repository, and place the two hdf5 data files from the Zenodo
repository in the `data` directory:

    $ git clone https://github.com/Jfeatherstone/tracker
    $ cd tracker
    $ curl <zenodo_link> --output ./zenodo_data.zip
    $ unzip zenodo_data.zip && mv zenodo_data/data .

You can now load and work with the data with the tools provided by this package.

    from tracker import loadAntData

    dataFile = '/path/to/ant_dataset_1.h5'

    # debug=True gives a nice progress bar and status messages
    dataArr, timeArr, metadataArr = loadAntData(dataFile, debug=True)

For more information, please see the documentation.
