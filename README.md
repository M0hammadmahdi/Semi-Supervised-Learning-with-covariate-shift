This file provides information about running the code to reproduce the results mentioned in the paper. Details of the sub-directories are mentioned below

- docs: contains the requirements.txt file needed to install all the libraries/packages required to execute the code.

- Experiment 1, Experiment 2: contain the source code and other dependencies required to run those experiments. 

Please use the command below to create a virtual environment with required packages. We assume that python3, virtualenv and pip3 are available. If not, please install them before executing the commands below.

1. virtualenv -p python3 virtual-env   # creates a virtual environment called virtual-env

2. source virtual-env/bin/activate

3. pip3 install -r docs/requirements.txt


For running the first experiment please run CSSL_synthetic.py within the virtual environment.
Similarly, for the second experiment please run CSSL_MNIST.py
