# 
# Import of generic libraries
# 
import time
import sys

# 
# Import of JAX libraries
# 
import jax
import jax.numpy as jnp

# 
# Other import
# 
import config
import lib.log as log
import lib.setup as setup
import lib.neural_network as neuralNetwork
import lib.loss_function as lossFunction
import lib.z2_valued_function as z2

# Test sanity of some data
if(len(sys.argv) < 2):
	raise Exception("No run name specified.") 

if(len(config.BRANCH_POINTS[0]) != len(config.BRANCH_POINTS[1])):
	raise Exception("Not all coordinates of the branch points are defined") 

if(len(config.BRANCH_POINTS[0]) % 2 == 0):
	raise Exception("As the north pole is also a branch point, there must be an odd number of points in BRANCH_POINTS") 

# 
# Show generic information
# 
jax.print_environment_info()

# 
# Create log file
# 
logLabels = ["Epoch", "Accuracy", "Overall Loss", "RMSE", "Maximal Error", "Min distance Branch Points", "C0-norm", "Eigenvalue", "Time (s)"]
logFile = log.openLogFile(logLabels)

# 
# Keep track when the last time the weights are saved
# 
lastTimeSaved = time.time()

# 
# Initialize the pseudorandom number generator
# 
globalKey = jax.random.key(config.RANDOM_SEED)

# 
# Generate the set of points where we measure our function
# 
print("Generate pointcloud")
globalKey, randomKey = jax.random.split(globalKey)
pointSet = setup.generatePointCloud(
	randomKey,
	config.BATCH_SIZE * config.BATCHES)

globalKey, randomKey = jax.random.split(globalKey)
validationSet = setup.generatePointCloud(
	randomKey,
	config.VALIDATION_SET_SIZE)

# 
# Generate the initial parameters
# 
print("Generate initial parameters")
print("Number of parameters: {}".format(setup.countParameters()))
print("Number of points: {}". format(config.BATCH_SIZE * config.BATCHES))
globalKey, randomKey = jax.random.split(globalKey)
parameters = setup.generateParameters(randomKey)

# Create a copy of the parameters (with only zeros) needed for the AdamW optimiser
parametersM =  setup.generateAdamParameters()
parametersV = setup.generateAdamParameters()

# 
# The training loop
# 
print("Start training")
for epoch in range(config.EPOCHS):
	startTime = time.time()
	statistics = []

	# Get learning rate for this round
	learningRate = neuralNetwork.cosineLearningRate(epoch)

	# Shuffling
	globalKey, randomKey = jax.random.split(globalKey)
	shuffled = jax.random.permutation(randomKey, pointSet, axis=1) 
	batches = jnp.split(shuffled, config.BATCHES, axis=1)

	# Mini batching
	for batch in batches:
		# 
		# This is code to check for nans
		# 
		if(config.DEBUG_NANS):
			print("phi: {}, theta: {}".format(parameters[4][0], parameters[4][1]))

			test1 = z2.Z2ValuedFunction(batch, parameters).block_until_ready()
			test2 = z2.Z2ValuedLaplacian(batch, parameters).block_until_ready()
			test3 = lossFunction.weightFunction(batch, parameters[4]).block_until_ready()

			nans1 = jnp.isnan(test1)	
			nans2 = jnp.isnan(test2)	
			nans3 = jnp.isnan(test3)	

			totalNans = jnp.sum(nans1) + jnp.sum(nans2) + jnp.sum(nans3)

			if(totalNans > 0):
				for i in range(config.BATCH_SIZE):
					if(nans1[i] or nans2[i] or nans3[i]):
						if(nans1[i]):
							print("FOUND NAN IN FUNCTION.")
						if(nans2[i]):
							print("FOUND NAN IN LAPLACIAN.")
						if(nans3[i]):
							print("FOUND NAN IN WEIGHT.")

						phi,theta = z2._toSpherical(jnp.asarray([batch[0][i], batch[1][i], batch[2][i]]))
						print("Coordinate: {}, {}, {}".format(batch[0][i], batch[1][i], batch[2][i]))
						print("position: {}, {}".format(phi,theta))

						branchPoints = parameters[4]
						num = len(branchPoints[0])
						for j in range(num):
							print("Branch point {}: {}, {}".format(j, branchPoints[0][j], branchPoints[1][j]))
							print("Distance from branch point {}: {}, {}".format(j, phi - branchPoints[0][j],theta - branchPoints[1][j]))

		parameters, parametersM, parametersV, loss = neuralNetwork.update(
			parameters, parametersM, parametersV,
			batch, learningRate, epoch)
		statistics.append(loss)

	statistics = jnp.asarray(statistics)
	meanStats = jnp.mean(statistics, axis=0).block_until_ready()
	stdStats = jnp.std(statistics, axis=0).block_until_ready()

	# Validation
	accuracy, unused = lossFunction.lossFunction(validationSet, parameters)
	accuracy = accuracy.block_until_ready()
	
	endTime = time.time()

	# Save data
	if  time.time() - lastTimeSaved > config.SAVE_EVERY:
		lastTimeSaved = time.time()

		log.saveParameters(epoch, parameters)
		print("-- Parameters Saved --\n")

	# Log to file
	log.writeToFile(logFile, epoch, accuracy, meanStats, stdStats, endTime - startTime)

	# Log to screen
	log.writeToScreen(logFile, logLabels, epoch, accuracy, meanStats, stdStats, endTime - startTime)

# Save last run seperately
log.saveParameters("last", parameters)
print("-- Parameters Saved --\n")

log.closeLogFile(logFile)
