# ObjectDetectionRL
This repo contains the final project for our Reinforcement Learning course. The project contains object detection
framework using a deep reinforcement learning agent. The core model is a DQN
whose actions transform a bounding box until it tightly bounds the desired object.
The code is inspired by this [work](https://github.com/rayansamy/Active-Object-Localization-Deep-Reinforcement-Learning).

## Setup

Create a new conda environment and download all the required packages using the following line

`conda env create --file=environment.yml`

To use an existing conda environment and simply update its packages for this repo, run

`conda env update -name <MYENV> --file environment.yml --prune`

Please note that this package installs cudatoolkit version 11.1, which may not be compatible with your GPU drivers.

## Data 

Download the data by running the `prep_dataset.sh` script provided
in the util folder. Run the script as follows

`cd util`

`bash prep_dataset.sh <FOLDER_TO_STORE_DATA>`

## Training

To train the agents, use the following command:

`python main.py --mode train --data-dir <DATA_FOLDER> --save-dir <SAVE_FOLDER>`
