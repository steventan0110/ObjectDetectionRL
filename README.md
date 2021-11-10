# ObjectDetectionRL
Final Project for Reinforcement Learning

## Setup

Create a new conda environment and download all the required packages using the following line

`conda env create --file=environment.yml`

To use an existing conda environment and simply update its packages for this repo, run

`conda activate <MYENV>`

`conda env update --file environment.yml --prune`

## Data 

Download the data by running the `prep_dataset.sh` script provided
in the util folder. Run the script as follows

`cd util`

`bash prep_dataset.sh <ROOT_FOLDER>`

