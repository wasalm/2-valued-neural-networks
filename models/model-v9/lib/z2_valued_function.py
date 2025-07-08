#
# We consider the S^2 (with radius 1) punctured by certain number of random even points.
# One point is always on the north pole
# To define the Z_2 valued function we consider a start-shaped graph G on S^2, and
# and we restrict our attention to the complement of this graph.
#
# We cover S^2 - G with open balls that are centered at each puncture. 
# We also at a chart at the south pole.
# On each open ball we define a Z_2 valued function that is supported on that ball.
# We sum all these functions to get a globally defined Z_2 valued function.
#
# To define such a function we start with a (random) neural network.
# We antisymmetrize and we multiply its result with a step function.
#

#
# When we equip a chart with spherical coordinates,
# the metric on the base space will be
# 	g = d phi^2 + sin^2 (phi) d theta^2

import os
import sys

import jax
import jax.numpy as jnp

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
#
# Import remaining files
#
import config
import lib.neural_network as neuralNetwork

#
# Smooth step up/down function f(r),
# that is 0/1 if r < a and 1/0 if r > b
# for 0 < a < b
# Input and output: number
#
def _stepUp(r,a,b):
	# Define the function f(s) = exp(-1/s), extended by zero
	# To define the derivative at zero properly we use the trick found in
	# https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
	s_fixed = lambda s: jnp.where(jnp.greater(s, 0.0), s, -1.0)
	f = lambda s: jnp.where(jnp.greater(s, 0.0), jnp.exp(-1.0/s_fixed(s)),0.0)
	
	# Next let g be the smooth step function
	# with a = 0 and b = 1
	g = lambda s: f(s) / (f(s) + f(1-s))

	# Next rescale r such that 
	# t < 0 when r < a and t > 1 when r > b
	t = (r - a)/(b - a)
	
	return g(t)

stepUp = jax.jit(_stepUp)

def _stepDown(r,a,b):
	return 1.0 - stepUp(r,a,b)

stepDown = jax.jit(_stepDown)

#
# Get polar coordinates
# N.B. The derivative of this function will yield a nan at the poles!
#
def _toSpherical(coord):
	# We calculate the angles phi and theta
	phi = jnp.acos(coord[2])
	theta = jnp.atan2(coord[1], coord[0])	

	# Force the range of theta between 0 and 2 pi
	theta = jnp.remainder(theta, 2 * jnp.pi)
	return phi, theta

toSpherical = jax.jit(_toSpherical)

# 
# Definition of a Z_2-valued function on,
# supported on a single chart
# This function is different if the chart is centered at a corner vertex
# or at the vertex in the centre of the graph.
# 
def southPoleChart(coord, branchPoints, weights):
	# We use the xy coordinates as input of the neural network
	result1 = neuralNetwork.evaluate(coord[:-1], weights)[0]

	# print("TODO: southPoleChart")
	# result1 = jnp.ones_like(coord[0])

	# Calculate boundary of the chart
	minPhi = jnp.max(branchPoints[0])

	# Convert to spherical coordinates, to calculate bounds of the chart
	phi, theta = toSpherical(coord)
	result2 = stepUp(
		phi,
		minPhi,
		jnp.pi)

	return result1 * result2

# N.B. The derivative of this function will yield a nan at the poles and on the graph
def northPoleChart(coord, branchPoints, weights):

	# Spherical coordiates
	phi, theta = toSpherical(coord)

	# We lift these coordinates to double cover
	liftedPhi = jnp.sqrt(phi)
	liftedTheta = jnp.remainder(theta -  branchPoints[1][0], 2.0 * jnp.pi) / 2.0
	liftedCoords = jnp.asarray([
		liftedPhi * jnp.cos(liftedTheta),
		liftedPhi * jnp.sin(liftedTheta)
	])

	# Consider an antisymmetric function on this double cover
	result1 = neuralNetwork.evaluate(liftedCoords, weights)[0]
	result1 -= neuralNetwork.evaluate(-liftedCoords, weights)[0]

	# Set correct sign at the other places
	sign = 1.0
	for angle in branchPoints[1][1:]:  # This is a small loop, so no big performance panalty
		sign *= jnp.where(
			jnp.greater(
				theta,
				jnp.remainder(angle, 2 * jnp.pi)),
			-1.0,
			1.0)
	result1 *= sign

	# Calculate boundary of the chart
	maxPhi = jnp.min(branchPoints[0])

	#Calculate bounds of the chart
	result2 = stepDown(
		phi,
		0.0,
		maxPhi)

	return result1 * result2 * phi

def genericChart(coord, branchPointsIndex, branchPoints, weights):
	# Change to spherical coordinates
	phi, theta = toSpherical(coord)

	# Search for the boundary of the phis
	phis = branchPoints[0]
	phi0 = phis[branchPointsIndex] # branching point of interest
	minPhi = jnp.min(phis)
	maxPhi = jnp.max(phis)

	# Search for the boundary of the thetas
	thetas = branchPoints[1]
	theta0 = thetas[branchPointsIndex]  # branching point of interest

	# Extend the list of remaining branch points with a factor of +- pi in order to catch the edge cases
	otherThetas = thetas - 2.0 * jnp.pi
	otherThetas = jnp.append(otherThetas, thetas[:branchPointsIndex])
	otherThetas = jnp.append(otherThetas, thetas[branchPointsIndex+1:])
	otherThetas = jnp.append(otherThetas, thetas + 2.0 * jnp.pi)

	minTheta = jnp.max(jnp.where(jnp.less(otherThetas, theta0), otherThetas, -jnp.inf))
	maxTheta = jnp.min(jnp.where(jnp.greater(otherThetas, theta0), otherThetas, jnp.inf))

	# put the range of theta such that it does not jump within the chart
	theta = jnp.remainder(theta - minTheta, 2.0 * jnp.pi) + minTheta

	# Step functions
	result2 = stepUp(
		phi,
		0.0,
		minPhi)

	result2 *= stepDown(
		phi,
		maxPhi,
		jnp.pi)

	result2 *= stepUp(
		theta,
		minTheta,
		theta0)

	result2 *= stepDown(
		theta,
		theta0,
		maxTheta)

	# Recenter the chart with the origin at the branch point
	phi = phi - phi0
	theta = theta - theta0

	angle = (jnp.atan2(theta, phi) + jnp.pi)/2

	innerProduct = \
		coord[0] * jnp.sin(phi0) * jnp.cos(theta0) + \
		coord[1] * jnp.sin(phi0) * jnp.sin(theta0) + \
		coord[2] * jnp.cos(phi0)

	#Issue: whem we get close to the antipodal point, the derivative of the arctan will behave badly.
	#Solution: Interpolate with a safe distance function

	step = stepUp(innerProduct, -0.5, 0)
	safeInnerProduct = (innerProduct + 0.5) * step - 0.5
	distance = jnp.acos(safeInnerProduct) + (1 - step) * jnp.sqrt(phi * phi + theta * theta)

	liftedCoords = jnp.asarray([
		jnp.sqrt(distance) * jnp.cos(angle),
		jnp.sqrt(distance) * jnp.sin(angle)
	])

	# Consider an antisymmetric function on this double cover
	result1 = neuralNetwork.evaluate(liftedCoords, weights)[0]
	result1 -= neuralNetwork.evaluate(-liftedCoords, weights)[0]

	# Mutiply result with antisymmetric function
	return result1 * result2 * distance

#
# Globally defined Z_2-valued function
#
def _Z2ValuedFunction(coord, parameters):
	northPoleNetwork, southPoleNetwork, networks, eigenvalueParameter, branchPoints = parameters

	# Normalize coordinate such that we create an r-invariant function in R^3
	# This is important when we calculate the gradient
	# Hence, this is NOT redundant
	coord = coord / jnp.linalg.vector_norm(coord,axis=0)

	result = southPoleChart(coord, branchPoints, southPoleNetwork)
	result += northPoleChart(coord, branchPoints, northPoleNetwork)

	# Remaining charts
	# N.B. We know that loops will be unrolled, 
	# But as we only loop over a few charts, is is not that bad

	num = len(branchPoints[0])
	for i in range(num):
		result += genericChart(coord, i, branchPoints, networks[i])

	return result

Z2ValuedFunction = jax.jit(_Z2ValuedFunction)

# 
# Bodge: the coordinate charts already asssume array input
# This will break when calculating the derivatives
# 
def singleZ2ValuedFunction(coord, parameters):
	x, y, z = coord
	return Z2ValuedFunction(
		jnp.asarray([[x],[y],[z]]),
		parameters)[0]

#
# Derivatives
# 
z2ValuedHessian = jax.jit(jax.vmap(
	jax.jit(jax.hessian(singleZ2ValuedFunction,0)),
	in_axes=(1,None)))

#
# Laplacian of the Z_2 valued function
# We use that the function can be seen as a radially invariant function in \R^3 - \{0\}.
# So we can take the standard laplacian in R^3
#
def _Z2ValuedLaplacian(coord, parameters):
	hessian = z2ValuedHessian(coord, parameters)
	result = - jnp.trace(hessian, axis1=1, axis2=2)
	return result

Z2ValuedLaplacian = jax.jit(_Z2ValuedLaplacian)
