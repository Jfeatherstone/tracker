#!/bin/bash

#SBATCH -p compute
#SBATCH -t 06:00:00
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --job-name=PreprocessImg
#SBATCH --mail-type ALL
#SBATCH --mail-user jack.featherstone@oist.jp

##############################################
# Script to preprocess all unprocessed datasets
# in the scan folder
##############################################

# Make sure the following variables are set correctly before running!

# The folder to search for datasets, and where to move the
# processed files after completion
dataFolder="/bucket/BandiU/Jack/data/scans/"
preprocessScript="/bucket/BandiU/Jack/python/web_scanner/scripts/preprocess_cli.py"
datasetStatusScript="/bucket/BandiU/Jack/python/web_scanner/scripts/dataset_status.py"
# Make sure this is matching the slurm settings above
numCores=32

# Not recommended to edit below this line unless you know what you're
# doing.
###############################################

echo "Data input/output directory: $dataFolder"
echo "Dataset status script: $datasetStatusScript"
echo "Preprocess script: $preprocessScript \n"

# create a temporary directory for this job and save the name
tempdir=$(mktemp -d /flash/BandiU/jack_temp.XXXXXX)

echo "Created temporary directory: $tempdir"

# enter the temporary directory
cd $tempdir

# Activate conda environment
source /home/j/john-featherstone/.zshrc
conda activate 311

# Output the processed datasets in the current folder (and then we will
# copy them to bucket after).
outputFolder="./"

# This will run the preprocessing for all datasets that are unprocessed
for dataset in $(python3 $datasetStatusScript $dataFolder --list-unprocessed); do
    echo $dataset
    python3 $preprocessScript -p $numCores -o $outputFolder $dataset
done

# copy our result back to Bucket. We use "scp" to copy the data 
# back as bucket isn't writable directly from the compute nodes.
scp -r ./* "deigo:$dataFolder"

# Clean up by removing our temporary directory
rm -r $tempdir
