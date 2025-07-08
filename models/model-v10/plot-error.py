import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import config
import lib.setup as setup
import lib.loss_function as loss
import lib.z2_valued_function as z2

import glob
import os
import pickle

#
# Generate data point
#

def generateDataPoints(parameters):

	# Generate coordinate grid
	phis = jnp.arange(jnp.pi/config.TEXTURE_HEIGHT/2.0, jnp.pi, jnp.pi/config.TEXTURE_HEIGHT)
	thetas = jnp.arange(jnp.pi/config.TEXTURE_WIDTH, 2.0*jnp.pi, 2.0*jnp.pi/config.TEXTURE_WIDTH)

	# Next we combine them into a grid
	phis, thetas = jnp.meshgrid(phis, thetas)

	# We need this variable to make the result in the same shape
	groupSize = len(phis) 

	flattenedPhis = jnp.concatenate(phis)
	flattenedThetas = jnp.concatenate(thetas)

	coords = jnp.asarray((
		jnp.sin(flattenedPhis) * jnp.cos(flattenedThetas),
		jnp.sin(flattenedPhis) * jnp.sin(flattenedThetas),
		jnp.cos(flattenedPhis)
	))


	# Next we calculate the error
	function, laplacian, eigenvalue, error = loss.errorFunction(coords, parameters)

	weightedError = jnp.sqrt(loss.weightFunction(coords, parameters[4])) * error

	# We normalize such that the sup norm of the function = 1
	normC0 = jnp.max(jnp.abs(function))

	print("C0 norm: {}".format(normC0))

	# error = error / normC0
	error = weightedError / normC0

	errorAbs = jnp.abs(error)
	lossC0 = jnp.max(errorAbs)
	
	print("C0 Error: {}".format(lossC0))

	error = jnp.array_split(error, groupSize)

	return phis, thetas, error

# Plotting functions
def plotTexture(data, branchPoints, path):
	fig, ax = plt.subplots(frameon=False, figsize=(2,1), dpi=config.TEXTURE_HEIGHT*4)
	ax = fig.add_axes([0, 0, 1, 1])
	ax.pcolormesh(data[1], data[0], data[2], cmap='RdBu_r', norm=colors.CenteredNorm())
	# ax.plot(branchPoints[1], branchPoints[0], '*', markersize=0.1, color="k")

	plt.xlim(0, 2 * jnp.pi-0.01)
	plt.ylim(0, jnp.pi)

	# Remove background
	for item in [fig, ax]:
	    item.patch.set_visible(False)

	# Remove axis
	fig.patch.set_visible(False)
	ax.axis('off')

	# Save
	fig.savefig(path)  

def plotColorMap(data, branchPoints, path):
	fig, ax = plt.subplots(figsize=(8,4))

	plt.xlim(0, 2 * jnp.pi-0.01)
	plt.ylim(0, jnp.pi)

	pc = ax.pcolormesh(data[1], data[0], data[2], cmap='RdBu_r', norm=colors.CenteredNorm())
	# ax.plot(branchPoints[1], branchPoints[0], '*', markersize=2.0, color="k")

	plt.xlabel("Azimuthal angle (θ)")
	plt.ylabel("Polar angle (φ)")
	
	fig.colorbar(pc, ax=ax)
	fig.savefig(path, dpi = 256)  

# 
# Main code
# 
listOfFileNames = glob.glob(config.SAVE_FOLDER + "/*/*.pickle") # * means all if need specific format then *.csv

fileName = max(listOfFileNames, key=os.path.getctime)
runName = fileName.split('/')[-2]

with open(fileName, 'rb') as file:
	parameters = pickle.load(file)

	branchPoints = setup.normalizeBranchPoints(parameters[4])
	data = generateDataPoints(parameters)
	plotTexture(data, branchPoints, config.SAVE_FOLDER + '/' + runName + '/texture-error-latest.jpg')
	plotColorMap(data, branchPoints, config.SAVE_FOLDER + '/' + runName + '/plot-error-latest.jpg')
