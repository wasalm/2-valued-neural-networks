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
import lib.z2_valued_function as z2

# 
# Calculate the distance to a branch point
# We also remove the antipodal points, as the derivative can also break here.
# 
def minimumDistanceToBranchPoints(coord, branchPoints):
	phi, theta = z2.toSpherical(coord)

	# Distance from north pole and south pole
	result = jnp.minimum(phi, jnp.pi - phi)

	# Distance from theta = 0
	if(config.SOUTHPOLE_IS_BRANCH_POINT):
		result = jnp.minimum(result, theta)
		result = jnp.minimum(result, 2.0* jnp.pi - theta)

	# Weight at other poles
	num = len(branchPoints[0])
	for i in range(num):
		phi0 = branchPoints[0][i]
		theta0 = branchPoints[1][i]

		diffTheta = jnp.remainder(theta - theta0 + jnp.pi, 2.0 * jnp.pi) - jnp.pi
		diffPhi = phi - phi0

		distance = jnp.sqrt(jnp.square(diffTheta) + jnp.square(diffPhi))
		result = jnp.minimum(result, distance)

	return result	

def minimumDistanceBetweenEdgeLines(branchPoints):
	result = jnp.inf

	num = len(branchPoints[0])

	for i in range(num):

		phi0 = branchPoints[0][i]
		theta0 = branchPoints[1][i]

		# distance to north pole
		result = jnp.minimum(result, phi0)

		# distance to south pole
		result = jnp.minimum(result, jnp.pi - phi0)

		for j in range(i+1, num):
			theta1 = branchPoints[1][j]
			distance = jnp.abs(jnp.remainder(theta0 - theta1 + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
			result = jnp.minimum(result, distance)

	return result

# 
# Define a weight function. We multiply the error with this weight function in the C^0 norm
# 
def weightFunction(coord, branchPoints):
	phi = jnp.acos(coord[2])

	# Weight at north pole
	step = z2.stepDown(phi, 0.1, 0.2)
	result = step * phi + (1.0-step)

	# Weight at south pole
	if(config.SOUTHPOLE_IS_BRANCH_POINT):
		step = z2.stepUp(phi, jnp.pi - 0.2, jnp.pi - 0.1)
		result *= step * (jnp.pi - phi) + (1.0-step)

	# Weight at other poles
	num = len(branchPoints[0])
	for i in range(num):
		# Calculate the XYZ coordinate of the branch point
		phi = branchPoints[0][i]
		theta = branchPoints[1][i]

		innerProduct = coord[0] * jnp.sin(phi) * jnp.cos(theta) + \
			coord[1] * jnp.sin(phi) * jnp.sin(theta) + \
			coord[2] * jnp.cos(phi)

		# Need to do a trick in order to prevent nans
		safeInnerProduct = jnp.where(jnp.less(innerProduct, -0.996), -0.996, innerProduct)
		distance = jnp.acos(safeInnerProduct)

		step = z2.stepUp(distance, 0.1, 0.5)
		result *= step + distance * (1.0 - step)

	return result


# 
# Define the error loss function, 
# This will be a combination of the L^2, L^1, C^0 estimate of \Delta u - \lambda u
# We also add the factor abs(1 - \|f\|^2) for normalization
# and the eigenvalue estimate
# 

def _errorFunction(coords, parameters):
	function = z2.Z2ValuedFunction(coords, parameters)
	laplacian = z2.Z2ValuedLaplacian(coords, parameters)

	eigenvalue = config.MINIMAL_EIGENVALUE + (config.MAXIMAL_EIGENVALUE - config.MINIMAL_EIGENVALUE) * (1.0 + jnp.tanh(parameters[3])) / 2.0
	error = laplacian - eigenvalue * function

	return function, laplacian, eigenvalue, error

errorFunction = jax.jit(_errorFunction)

def _unsafeLossFunction(coords, parameters, factor):
	function, laplacian, eigenvalue, error = errorFunction(coords, parameters)

	error = error * factor # Set error to zero near the graph

	weightFct = weightFunction(coords, parameters[4])
	weightedError = jnp.sqrt(weightFct) * error

	# We normalize such that the sup norm of the function = 1
	normC0 = jnp.max(jnp.abs(function))

	lossL2 = jnp.sqrt(jnp.average(jnp.square(error)))
	lossC0 = jnp.max(jnp.abs(weightedError))
	lossNorm = jnp.square(config.C0_NORMALIZATION - normC0)

	# Penalize if the branch points are getting too close.
	distance = minimumDistanceBetweenEdgeLines(parameters[4])
	step = z2.stepDown(distance, 0.5 * config.MINIMAL_BRANCH_DISTANCE, config.MINIMAL_BRANCH_DISTANCE) 
	lossBranchDistance = (1.0 / distance - 1.0 / config.MINIMAL_BRANCH_DISTANCE) * step

	loss = config.LOSS_L2 * lossL2 + \
			config.LOSS_C0 * lossC0 + \
			config.LOSS_NORM * lossNorm + \
			config.LOSS_EIGENVALUE * eigenvalue + \
			config.LOSS_BRANCH_DISTANCE * lossBranchDistance

	return loss, jnp.asarray([loss, lossL2, lossC0, distance, normC0, eigenvalue])

unsafeLossFunction = jax.jit(_unsafeLossFunction)

# The loss function may return NaN (Not a Number) when it is evaluated at a branch point.
# Hence we filter out these cases and set the loss to zero at these points.

def _lossFunction(coords, parameters):
	branchPoints = parameters[4]

	# 
	# First we create a safe point to evaluate in case of error
	# (Trick in Jax)
	# 
	sortedPhis = jnp.sort(branchPoints[0])
	sortedThetas = jnp.sort(branchPoints[1])
	phi0 = jnp.full_like(coords[0], (sortedPhis[0] + sortedPhis[1])/2)
	theta0 = jnp.full_like(coords[0], (sortedThetas[0] + sortedThetas[1])/2)

	exclusionDistance = jnp.full_like(coords[0], config.EXCLUSION_DISTANCE) 

	safePoint = jnp.asarray([
		jnp.sin(phi0) * jnp.cos(theta0),
		jnp.sin(phi0) * jnp.sin(theta0),
		jnp.cos(phi0)
	])

	distance = minimumDistanceToBranchPoints(coords, parameters[4])
	distance = jnp.greater(distance, exclusionDistance)
	safeCoords = jnp.where(distance, coords, safePoint)

	result = unsafeLossFunction(safeCoords, parameters, distance.astype(int))
	return result

	
lossFunction = jax.jit(_lossFunction)
valAndGradLossFunction = jax.jit(jax.value_and_grad(lossFunction, argnums=1, has_aux=True))
