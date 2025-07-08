# Install script
Download the code using GIT

	git clone git@git.andries-salm.com:mathematics/ai-for-z2-eigenfunctions.git
	cd ai-for-z2-eigenfunctions

## Supercomputer
On the ULB supercomputer, we need to run the following code to enable Python:

	module purge
	module load releases/2025a
	module load Python/3.13.1-GCCcore-14.2.0

## Generic setup
With the following lines we setup the virtual environment and install the needed libraries:

	python3 -m venv --system-site-packages ./environment
	source ./environment/bin/activate
	python -m pip install -U pip
	python -m pip install numpy wheel
	python -m pip install matplotlib

## Installation of jax
Depending on machine, we need to install a different version of jax.

### MacOS
If we run on MacOS and we want to use the experimental driver, we run

	python -m pip install jax-metal

### A computer with a graphics card (CUDA)
If we have a graphics card available that supports CUDA, we run

	python -m pip install "jax[cuda12]"

### Other cases
In any other case we run

	python -m pip install jax

## Running of code
To run the code, we run

	source ./environment/bin/activate
	python main.py