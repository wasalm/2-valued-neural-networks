# 2-valued-neural-networks
Source code for the paper "Search for Z/2 eigenfunctions on the sphere using machine learning"

In the folder /models you can find the last two versions of our source code.
In the folder /web you can view the statistics of our most interesting runs.
It also has some 3d renders of our Z/2 eigenfunction.

# Quick links

Statistics of our runs: https://wasalm.github.io/2-valued-neural-networks/
3d Render of in the
Tetrehedral case: https://wasalm.github.io/2-valued-neural-networks/sphere.html?new-example
Squashed tetrahedral case: https://wasalm.github.io/2-valued-neural-networks/sphere.html?tetrahedra
Cubical case: https://wasalm.github.io/2-valued-neural-networks/sphere.html?cube


# Install script
Download the code using GIT

	git clone https://github.com/wasalm/2-valued-neural-networks
	2-valued-neural-networks

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
	./train-local.sh
