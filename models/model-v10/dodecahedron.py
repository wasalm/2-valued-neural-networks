import jax
import jax.numpy as jnp

# 
# This is a simple script to generate the coordinates of a dodecahedron
# 

phi = (1.0 + jnp.sqrt(5.0))/ 2.0
phiInv = 1.0 / phi

def _toSpherical(coord):
	phi = (1.0 + jnp.sqrt(5.0))/ 2.0
	phiInv = 1.0 / phi
	polarAxis = jnp.asarray([1,1,1]) / jnp.sqrt(3)
	azimuthReference = jnp.asarray([phiInv, 0, phi]) / jnp.sqrt(3)

	coord = coord/jnp.sqrt(jnp.dot(coord, coord))

	# We create a reference coordinate system
	# The z-axis aligned to a ramnification point
	zAxis = polarAxis/jnp.sqrt(jnp.dot(polarAxis, polarAxis))

	# The x-axis is the orthogonal projection of another ramnification point, ...
	xAxis = azimuthReference - jnp.dot(azimuthReference, zAxis) * zAxis

	# ..., that is normalized to unit length
	xAxis = xAxis/jnp.sqrt(jnp.dot(xAxis, xAxis))
	
	# Finally the y-axis is found using the cross product
	yAxis = jnp.cross(zAxis,xAxis)

	#We apply a basis transformation
	coordX = jnp.dot(xAxis, coord)
	coordY = jnp.dot(yAxis, coord)
	coordZ = jnp.dot(zAxis, coord)

	# We calculate the angles phi and theta
	phi = jnp.acos(coordZ)
	theta = jnp.atan2(coordY, coordX)	

	#We add this shift of 0.25 in order to prevent the overlap of the graph at zero.
	theta = jnp.remainder(theta + 0.25, 2 * jnp.pi) 
	return jnp.array([phi,theta])

toSpherical = jax.vmap(_toSpherical)

genericVertices = jnp.array([
	# ( 1, 1, 1),

	( phiInv, 0, phi),
	( phi, phiInv, 0),
	( 0, phi, phiInv),

	( 1,-1, 1),
	( phi,-phiInv, 0),
	( 1, 1,-1),
	( 0, phi,-phiInv),
	(-1, 1, 1),
	(-phiInv, 0, phi),

	( 0,-phi, phiInv),
	( 1,-1,-1),
	( phiInv, 0,-phi),
	(-1, 1,-1),
	(-phi, phiInv, 0),
	(-1,-1, 1),

	( 0,-phi,-phiInv),
	(-phiInv, 0,-phi),
	(-phi,-phiInv, 0),
	
	# (-1,-1,-1),
]) / jnp.sqrt(3)

sphericalCoordinates = jnp.transpose(toSpherical(genericVertices))