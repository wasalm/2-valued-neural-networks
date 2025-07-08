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
import config
import lib.loss_function as lossFunction
import lib.setup as setup


# 
# Learning rate schuduler
# 
def _cosineLearningRate(epoch):
	totalEpochs = config.RESTART_LEARNING_RATE_EVERY
	epoch = jnp.remainder(epoch, totalEpochs)

	minLearningRate = config.MIN_LEARNING_RATE
	maxLearningRate = config.MAX_LEARNING_RATE
	diff = maxLearningRate - minLearningRate

	return minLearningRate + diff * (
		1.0 + jnp.cos(jnp.pi * epoch / totalEpochs)
	) / 2.0

cosineLearningRate = jax.jit(_cosineLearningRate)

# 
# AdamW Optimiser
# See https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
# 
def _adamOptimiserM(m, gradient):
	beta1 = config.ADAM_BETA1
	return beta1 * m + (1.0 - beta1) * gradient

adamOptimiserM = jax.jit(_adamOptimiserM)

def _adamOptimiserV(v, gradient):
	beta2 = config.ADAM_BETA2
	return beta2 * v + (1.0 - beta2) * jnp.square(gradient)

adamOptimiserV = jax.jit(_adamOptimiserV)

def _adamWOptimiser(weight, newM, newV, gradient, learningRate, power):
	mHat = newM / (1.0 - jnp.pow(config.ADAM_BETA1, power))
	vHat = newV / (1.0 - jnp.pow(config.ADAM_BETA2, power))

	return (1.0 - learningRate * config.ADAM_WEIGHT_DECAY) * weight \
		- learningRate * mHat / (config.ADAM_EPSILON + jnp.sqrt(vHat))

adamWOptimiser = jax.jit(_adamWOptimiser)

# 
# Batch loop
# 
def _update(parameters, parametersM, parametersV, batch, learningRate, epoch):
	loss, gradient = lossFunction.valAndGradLossFunction(batch, parameters)

	newM = jax.tree.map(adamOptimiserM, parametersM, gradient)
	newV = jax.tree.map(adamOptimiserV, parametersV, gradient)

	newP0, newP1, newP2, newP3, newP4 = jax.tree.map(
		lambda w, m, v, g: adamWOptimiser(
			w, m, v, g, 
			learningRate, epoch+1),
		parameters, newM, newV, gradient)

	# Normalize branch points
	newP4 = setup.normalizeBranchPoints(newP4)

	# revert parameters if they are not trainable
	if not config.EIGENVALUE_TRAINABLE:
		newP3 = parameters[3]
		# Ignore the adam parameters

	if not config.BRANCHING_POINTS_TRAINABLE:
		newP4 = parameters[4]
		# Ignore the adam parameters

	newP = (newP0, newP1, newP2, newP3, newP4)
	
	return newP, newM, newV, loss[1]

update = jax.jit(_update)

#
# Run the neural network forwards
#
def _evaluate(coordinate, parameters):
	activationFunction = config.ACTIVATION_FUNCTION
	
	activations = coordinate

	for weight, bias in parameters[:-1]:
		outputs = jnp.matmul(weight, activations) + bias
		activations = activationFunction(outputs)

	finalWeight, finalBias = parameters[-1]
	return jnp.matmul(finalWeight, activations) + finalBias
evaluate = jax.jit(_evaluate)
