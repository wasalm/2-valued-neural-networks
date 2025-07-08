# 
# Import of generic libraries
# 
import glob
import os
import pickle

# 
# Import of JAX libraries
# 
import jax
import jax.numpy as jnp

# 
# Other import
# 
import config
import lib.setup as setup
import lib.loss_function as loss
import lib.z2_valued_function as z2

def testNeuralNetwork(file, log):
	# 
	# Initialize the pseudorandom number generator
	# 
	globalKey = jax.random.key(config.RANDOM_SEED_TEST)

	# 
	# Generate the set of points where we measure our function
	# 
	globalKey, randomKey = jax.random.split(globalKey)
	testingSet = setup.generatePointCloud(
		randomKey,
		config.TEST_SET_SIZE)

	# 
	# Load parameters
	# 
	parameters = pickle.load(file)
	

	# Next we calculate the error
	function, laplacian, eigenvalue, error = loss.errorFunction(testingSet, parameters)

	weightFct = loss.weightFunction(testingSet, parameters[4])
	weightedError = jnp.sqrt(weightFct) * error

	# 
	# Next we calulate the norms
	# 
	normC0 = jnp.max(jnp.abs(function))
	normL2 = jnp.average(jnp.square(function))

	errorC0 = jnp.abs(error) / normC0
	weightedErrorC0 = jnp.abs(weightedError) / normC0
	errorL2 = jnp.square(error) / normL2
	
	#
	# We also calculate the Reyleigh quotient
	normL21 = jnp.average(laplacian * function)
	rayleighQuotient = normL21 / normL2

	log.write("\t{}: {}\n".format("Eigenvalue", eigenvalue))
	log.write("\t{}: {}\n".format("Rayleigh quotient", rayleighQuotient))
	log.write("\t{}: {}\n".format("Root Mean Squared Error", jnp.sqrt(jnp.average(errorL2))))
	log.write("\t{}: {}\n".format("Maximal absolute Error", jnp.max(errorC0)))
	log.write("\t{}: {}\n".format("Weighted maximal absolute Error", jnp.max(weightedErrorC0)))

# 
# Main code
# 
listOfFileNames = glob.glob(config.SAVE_FOLDER + "/*/*.pickle") # * means all if need specific format then *.csv

fileName = max(listOfFileNames, key=os.path.getctime)
runName = fileName.split('/')[-2]

logFileName = config.SAVE_FOLDER + '/' + runName + '/test-results.txt'

with open(fileName, 'rb') as file:
	with open(logFileName, 'w') as log:
		log.write("Testing the Neural network\n\n")
		log.write("\tFilename: " + fileName + "\n")

		testNeuralNetwork(file, log)