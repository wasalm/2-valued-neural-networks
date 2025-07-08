import os
import sys

import datetime
import pickle

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

def openLogFile(labels):
	fileName = config.SAVE_FOLDER + "/" + sys.argv[1] + "/log.txt"
	logFile = open(fileName, "w")

	for label in labels[:-1]:
		logFile.write(label)
		logFile.write(", ")

	logFile.write(labels[-1])
	logFile.write("\n")
	logFile.flush()

	return logFile

def writeToFile(file, epoch, accuracy, mean, std, time):
	file.write("{}, ".format(epoch))
	file.write("{}, ".format(accuracy))

	for i in range(len(mean)):
		file.write("{} +/- {}, ".format(mean[i], std[i]))
	
	file.write("{}\n".format(time))
	file.flush()

def writeToScreen(file, labels, epoch, accuracy, mean, std, time):
	print("Epoch {} of {}".format(epoch+1, config.EPOCHS))
	print("\tAccuracy: {}".format(accuracy))

	for i in range(len(mean)):
		print("\t{}: {} +/- {}".format(labels[i+2], mean[i], std[i]))
		
	print("\tTime (sec): {:0.1f}".format(time))
	print("", flush=True)

def closeLogFile(file):
	file.close()

def saveParameters(epoch, parameters):
	path = "{}/{}/run-{}-{}.pickle".format(
		config.SAVE_FOLDER, 
		sys.argv[1],
		datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
		epoch,
	)

	with open(path, 'wb') as file:
		pickle.dump(parameters, file)
