import os
import sys

import jax
import jax.numpy as jnp
import jax.random as random

# 
# Add parent path in order to load the config file
# 
sys.path.append(
	os.path.dirname(
		os.path.dirname(
			os.path.realpath(__file__)
			)
		)
	)
import config

# 
# Normalize the branch points such that the range for phi is between 0 and pi and for theta between 0 and 2 pi
# 

def normalizeBranchPoints(branchPoints):	
	phis = branchPoints[0]
	thetas = branchPoints[1]

	# Set the range of phi between 0 and 2 pi
	phis = jnp.remainder(phis, 2 * jnp.pi)

	# If phi > pi 
	thetas = jnp.where(jnp.greater(phis, jnp.pi), thetas + jnp.pi, thetas)
	phis = jnp.where(jnp.greater(phis, jnp.pi), 2.0 * jnp.pi - phis, phis)

	# Normalize theta between 0 and 2 pi
	thetas = jnp.remainder(thetas, 2 * jnp.pi)
	return jnp.asarray([phis, thetas])

# 
# Define a set of points on the 2-sphere
# over which we can approximate the integration
# 
def generatePointCloud(randomnessKey, numberOfPoints):
	# N.B. this gives a uniform distribution over a ball. This is rotaionally symmetric,
	# So we can just divide out the norm to get a uniform norm on the 2-sphere
	points = jax.random.ball(
		randomnessKey,
		3,
		shape=numberOfPoints).transpose()

	length = jnp.linalg.vecdot(points, points, axis=0)
	return points / jnp.sqrt(length)

#
# Generate random parameters for a single layer
# We will use the initialisation methods described in https://arxiv.org/abs/1502.01852
# That is, we initialise the bias to zero, and pick the weights randomly from
# N(0,sqrt(2/n(l-1))) where n(l) is the number of neurons in the l'th layer
#
def generateLayerWithRandom(n,m, randomnessKey):
	return jnp.sqrt(2.0/m) * random.uniform(randomnessKey, shape=(n, m)), jnp.zeros((n,1))
	
def generateLayerWithZeros(n,m):
	return jnp.zeros((n, m)), jnp.zeros((n,1))

def generateNetworkParametersWithRandom(hiddenLayerSizes, randomnessKey):
	layerSizes = [2] + hiddenLayerSizes + [1]

	subKeys = random.split(randomnessKey, len(layerSizes))
	return [generateLayerWithRandom(n, m, k) for m, n, k in zip(layerSizes[:-1], layerSizes[1:], subKeys)] 

def generateNetworkParametersWithZeros(hiddenLayerSizes):
	layerSizes = [2] + hiddenLayerSizes + [1]

	return [generateLayerWithZeros(n, m) for m, n in zip(layerSizes[:-1], layerSizes[1:])] 

def generateParameters(randomnessKey):
	numBranchPoints = len(config.BRANCH_POINTS[0])
	keys = jax.random.split(randomnessKey, 2 + numBranchPoints)

	northPoleNetwork = generateNetworkParametersWithRandom(config.HIDDEN_LAYER_SIZE_NORTH_POLE, keys[-1])
	southPoleNetwork = generateNetworkParametersWithRandom(config.HIDDEN_LAYER_SIZE_SOUTH_POLE, keys[-2])

	networks = []
	for i in range(numBranchPoints):
		network = generateNetworkParametersWithRandom(config.HIDDEN_LAYER_SIZE_BRANCH_POINT, keys[i])
		networks.append(network)
		
	eigenvalueParameter = jnp.atanh((2.0 * (config.INITIAL_EIGENVALUE - config.MINIMAL_EIGENVALUE) / (config.MAXIMAL_EIGENVALUE - config.MINIMAL_EIGENVALUE)) - 1.0)
	branchPoints = normalizeBranchPoints(config.BRANCH_POINTS)
	return northPoleNetwork, southPoleNetwork, networks, eigenvalueParameter, branchPoints

def generateAdamParameters():
	numBranchPoints = len(config.BRANCH_POINTS[0])

	northPoleNetwork = generateNetworkParametersWithZeros(config.HIDDEN_LAYER_SIZE_NORTH_POLE)
	southPoleNetwork = generateNetworkParametersWithZeros(config.HIDDEN_LAYER_SIZE_SOUTH_POLE)

	networks = []
	for i in range(numBranchPoints):
		network = generateNetworkParametersWithZeros(config.HIDDEN_LAYER_SIZE_BRANCH_POINT)
		networks.append(network)
		
	eigenvalueParameter = 0.0
	branchPoints = jnp.zeros_like(config.BRANCH_POINTS)
	return northPoleNetwork, southPoleNetwork, networks, eigenvalueParameter, branchPoints


#
# Count the number of parameters in the model
#
def countParametersInNetwork(hiddenLayerSizes):
	numParameters = 0
	layerSizes = [2] + hiddenLayerSizes + [1]

	for m,n in zip(layerSizes[:-1], layerSizes[1:]):
		numParameters += m * n + n

	return numParameters

def countParameters():
	numParameters = 0
	numBranchPoints = len(config.BRANCH_POINTS[0])

	numParameters += countParametersInNetwork(config.HIDDEN_LAYER_SIZE_NORTH_POLE)
	numParameters += countParametersInNetwork(config.HIDDEN_LAYER_SIZE_SOUTH_POLE)
	numParameters += numBranchPoints * countParametersInNetwork(config.HIDDEN_LAYER_SIZE_BRANCH_POINT)

	if(config.EIGENVALUE_TRAINABLE):
		numParameters += 1

	if(config.BRANCHING_POINTS_TRAINABLE):
		numParameters += 2 * numBranchPoints

	return numParameters