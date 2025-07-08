import jax
import jax.numpy as jnp

#
# Configuration of the branching points
# In this case, we always assume that at the north pole is a branching point
# We also assume that there are other (odd number of) branching points.
# Finally we assume that there is no branching point on the south pole
# We describe the position of the branching points using Spherical coordinates [phi[], theta[]]
#

BRANCH_POINTS = jnp.asarray([
	[jnp.arccos(-1.0/3.0), jnp.arccos(-1.0/3.0), jnp.arccos(-1.0/3.0)],
	[0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0]
])

# BRANCH_POINTS = jnp.asarray([
# 	[jnp.pi / 2.0, jnp.pi / 2.0, jnp.pi / 2.0],
# 	[0.0,          jnp.pi / 3.0, 2 * jnp.pi / 3.0]
# ])

# BRANCH_POINTS = jnp.asarray([
#       [jnp.pi / 2.0, jnp.pi / 2.0,       jnp.pi / 2.0,       jnp.pi / 2.0,       jnp.pi / 2.0],
#       [0.0,          jnp.pi / 5.0, 2.0 * jnp.pi / 5.0, 3.0 * jnp.pi / 5.0, 4.0 * jnp.pi / 5.0]
# ])


# Are the branching points allowed to move during training?
BRANCHING_POINTS_TRAINABLE = False
MINIMAL_BRANCH_DISTANCE = 0.5

# 
# How close may we sample near the branch points?
# Smaller yields better results, but has the danger it gives a division by zero error
# 
EXCLUSION_DISTANCE = 0.005

# 
# We cover the sphere with charts and we define a deep neural network for each
# There is a chart for each of the poles and a chart around the remaining branching points.
# Here we define the hidden layers
# 

HIDDEN_LAYER_SIZE_BRANCH_POINT = [64, 64, 64]
HIDDEN_LAYER_SIZE_NORTH_POLE = [64, 64, 64]
HIDDEN_LAYER_SIZE_SOUTH_POLE = [64, 64, 64]

# Activation to use in the neural networks
ACTIVATION_FUNCTION = jax.nn.gelu

#
# Settings for the loss function
#

# Give a range of eigenvalues to search
# IF eigenvalue is NOT trainable, then it is fixed as the average of the min and max.
MINIMAL_EIGENVALUE = 0.0
INITIAL_EIGENVALUE = 5.0
MAXIMAL_EIGENVALUE = 100.0
EIGENVALUE_TRAINABLE = True

# As this problem we have to normalize our function. Here we set the C^0 norm of our function.
C0_NORMALIZATION = 1.0

# 
# Relative factors in the loss function
# Namely, the loss function is a linear combination of simpler loss functions.
# Here we determine the relative factor of each.
# 
LOSS_L2 = 10.0
LOSS_C0 = 1.0
LOSS_BRANCH_DISTANCE = 5.0
LOSS_NORM = 100.0
LOSS_EIGENVALUE = 2.0

#
# At how many points do we measure
# - per batch?
# - in the validation step?
#
BATCH_SIZE = 2048
VALIDATION_SET_SIZE = 8192
TEST_SET_SIZE = 16384

# 
# How many training rounds do we have
# and how many batches do we want per training round?
# 
EPOCHS = 5000
BATCHES = 256

# 
# Seed value for the pseudorandom number generator
# 
RANDOM_SEED = 1338
RANDOM_SEED_TEST = 260493

# 
# Learning scheduler parameters
# 
MIN_LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.001
RESTART_LEARNING_RATE_EVERY = 1000

# 
# Parameters for the optimiser (AdamW)
# 
ADAM_WEIGHT_DECAY = 0.004
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-08

# 
# Data collection
# 
SAVE_FOLDER = "./public/data"
SAVE_EVERY = 300 # in seconds

# 
# Plot settings
# 
TEXTURE_WIDTH = 512
TEXTURE_HEIGHT = 256

# 
# DEBUG SETTINGS
# 
DEBUG_NANS = False
