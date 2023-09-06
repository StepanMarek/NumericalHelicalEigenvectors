using LinearAlgebra;
using Printf;

module Model

using LinearAlgebra;
export getHamiltonian, getSiteProjection, getSeparableTransformation, getOverlap, getSigmaGamma;

function getHamiltonian(angle=0.1, totalDimension=4, extraAngle=0.0)
	# Determines the 2 site Hamiltonian - both sp1
	# Constructs weak coupling to neighbouring opposite polarity orbitals - the tube model
	hamiltonian=zeros(Float64, (totalDimension, totalDimension));
	# Initial boundary
	hamiltonian[1,3] = -cos(angle);
	hamiltonian[1,4] = sin(angle);
	#hamiltonian[1,4] = -sin(angle);
	hamiltonian[3,1] = -cos(angle);
	hamiltonian[4,1] = sin(angle);
	#hamiltonian[4,1] = -sin(angle);
	hamiltonian[2,4] = -cos(angle);
	hamiltonian[2,3] = -sin(angle);
	hamiltonian[4,2] = -cos(angle);
	hamiltonian[3,2] = -sin(angle);
	# For each index besides the boundary, do both forward and backwards coupling, otherwise do special coupling
	if 3 < totalDimension-2
		# otherwise only get boundaries
		for orbIndex in 3:(totalDimension-2)
			# Forward same polarisation
			hamiltonian[orbIndex, orbIndex+2] = -cos(angle);
			# Backwards same polarisation
			hamiltonian[orbIndex, orbIndex-2] = -cos(angle);
			# Different polarisation
			if orbIndex % 2 == 1
				hamiltonian[orbIndex, orbIndex+3] = sin(angle);
				#hamiltonian[orbIndex, orbIndex+3] = -sin(angle);
				hamiltonian[orbIndex, orbIndex-1] = -sin(angle);
			else
				hamiltonian[orbIndex, orbIndex+1] = -sin(angle);
				hamiltonian[orbIndex, orbIndex-3] = sin(angle);
				#hamiltonian[orbIndex, orbIndex-3] = -sin(angle);
			end
		end
	end
	# Final boundary
	hamiltonian[totalDimension-3,totalDimension-1] = -cos(angle+extraAngle)
	hamiltonian[totalDimension-2,totalDimension-1] = -sin(angle+extraAngle)
	hamiltonian[totalDimension-1,totalDimension-3] = -cos(angle+extraAngle)
	hamiltonian[totalDimension-1,totalDimension-2] = -sin(angle+extraAngle)
	hamiltonian[totalDimension-2,totalDimension] = -cos(angle+extraAngle)
	hamiltonian[totalDimension-3,totalDimension] = sin(angle+extraAngle)
	#hamiltonian[totalDimension-3,totalDimension] = -sin(angle)
	hamiltonian[totalDimension,totalDimension-2] = -cos(angle+extraAngle)
	hamiltonian[totalDimension,totalDimension-3] = sin(angle+extraAngle)
	#hamiltonian[totalDimension,totalDimension-3] = -sin(angle)
	return hamiltonian;
end

function getOverlap(tToS=0.1, angle=0.1, totalDimension=4, extraAngle=0.0)
	return tToS * getHamiltonian(angle, totalDimension, extraAngle) + I
end

function getSeparableTransformation(totalDimension=4)
	# gives the matrix for separable transformation
	transformation = zeros(Float64, (totalDimension, totalDimension))
	halfDimension = totalDimension ÷ 2
	for i in 1:halfDimension
		# First, get the x-component
		transformation[i, 2 * i - 1] = 1
		# Then, get the y-component
		transformation[halfDimension + i, 2*i] = 1
	end
	return transformation
end

function getSiteProjection(siteIndex, totalDimension=4)
	# Get the matrix that projects the eigenvectors into onsite polarisation vector
	projectionMatrix=zeros(Float64, (2, totalDimension))
	# Given in the position representation - need 1 + 2*(siteIndex - 1) = 2*(siteIndex) - 1 and 2*siteIndex
	projectionMatrix[1, 2*siteIndex - 1] = 1
	projectionMatrix[2, 2*siteIndex] = 1
	return projectionMatrix;
end

function getSigmaGamma(imagPart, totalDimension=4)
	# Basically zero everywhere, applied constatn imaginary part at x and y for first and last orbitals
	sigma = zeros(ComplexF64, (totalDimension, totalDimension))
	gamma = zeros(ComplexF64, (totalDimension, totalDimension))
	sigma[1,1] = imagPart * im
	sigma[2,2] = imagPart * im
	sigma[totalDimension-1,totalDimension-1] = imagPart * im
	sigma[totalDimension,totalDimension] = imagPart * im
	gamma[1,1] = imagPart
	gamma[2,2] = imagPart
	return sigma, gamma
end

end

using .Model;

dimension = 16
#vs = LinRange(0, 2, 1000)
#angles = LinRange(0, 0.9, 101)
angles = LinRange(0, 0.8, 21)
tToS = 0.001
imagPart = 0.1
eta = 1e-10
Es = LinRange(-3,3,5000)
#onsitePenalties = LinRange(0.0,0.1,10)
#onsitePenalties = LinRange(0, 1.0, 101)
onsitePenalties = LinRange(0.0, 1.0, 21)
#onsitePenalty = 0.1
#xis = LinRange(0.0, 2*pi, 201)
xis = [0.0]
targetDir = "data_run_5/"
greensIndexPairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
polarisationIndexPairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
orbitalIndices = [7, 8, 9, 10]
transform = getSeparableTransformation(dimension)
for vIndex in 1:length(angles)
	v = angles[vIndex]
	for onsiteIndex in 1:length(onsitePenalties)
		onsitePenalty = onsitePenalties[onsiteIndex]
		extraAngle = xis[1]
		hamiltonian = getHamiltonian(v, dimension, extraAngle)
		overlap = getOverlap(tToS, v, dimension, extraAngle)
		sigma, gamma = getSigmaGamma(imagPart, dimension)
		# display(transform * hamiltonian * transpose(transform))
		# add boundary conditions by adding very high on-site energy to certain orbitals
		# First y-orbital
		hamiltonian[2,2] = onsitePenalty
		# Last y-orbital
		# At no angle, this is always linearly polarised
		hamiltonian[dimension,dimension] = onsitePenalty
		# Determine the eigenvectors of the Hamiltonian in the position representation
		# Solve generalized eigenvalue problem
		factorisation = eigen(hamiltonian, overlap)
		eigenvectors = factorisation.vectors
		eigenvalues = factorisation.values
		transformedHamiltonian = adjoint(eigenvectors) * hamiltonian * eigenvectors
		transformedSigma = adjoint(eigenvectors) * sigma * eigenvectors
		transformedGamma = adjoint(eigenvectors) * gamma * eigenvectors
		# Open the relevant file in the target dir
		filename = @sprintf("%sangle_%5.3f_onsite_%5.3f", targetDir, v, onsitePenalty)
		file = open(filename * ".greens", "w+")
		# Write generic headers
		write(file, "#E")
		for indexPair in greensIndexPairs
			write(file, ",G<" * string(indexPair[1]) * string(indexPair[2]))
		end
		write(file, "\n")
		# Write specific headers
		write(file, @sprintf("#angle=%g,onsite=%g,tToS=%g,imagPart=%g,eta=%g\n", v, onsitePenalty, tToS, imagPart, eta))
		for E in Es
			Gret = inv((E+im*eta)*I - transformedHamiltonian - transformedSigma)
			Gless = inv(adjoint(eigenvectors)) * (im * Gret * transformedGamma * adjoint(Gret)) * inv(eigenvectors)
			write(file, @sprintf("%g", E))
			for indexPair in greensIndexPairs
				write(file, @sprintf(" %g %g", real(Gless[indexPair[1],indexPair[2]] - Gless[indexPair[2],indexPair[1]]), imag(Gless[indexPair[1],indexPair[2]] - Gless[indexPair[2],indexPair[1]])))
			end
			write(file, "\n")
		end
		close(file)
		# Done with the Green's function output
		# Continue by output of polarisation vectors and helicity
		for orbitalIndex in orbitalIndices
			# Each orbital is in a separate file
			file = open(filename * "_orb_" * string(orbitalIndex) * ".orbital", "w+")
			# Now, inside the file, output polarisation at each site and helicity at each cite (except the last site, where it is taken same as the last but one site)
			# Generic headers
			write(file, "#Pair/site index,p_x,p_y,h\n")
			# Output energy in header
			write(file, @sprintf("#energy=%g,angle=%g,onsite=%g,tToS=%g,imagPart=%g,eta=%g\n", eigenvalues[orbitalIndex], v, onsitePenalty, tToS, imagPart, eta))
			currentPol = zeros(Float64, (2,1))
			nextPol = zeros(Float64, (2,1))
			currentPolRot = zeros(Float64, (2,1))
			nextPolRot = zeros(Float64, (2,1))
			helicity = 0.0
			for siteIndex in 1:(length(polarisationIndexPairs)-2)
				currentPair = polarisationIndexPairs[siteIndex]
				nextPair = polarisationIndexPairs[siteIndex+1]
				currentPol[1] = eigenvectors[currentPair[1], orbitalIndex]
				currentPol[2] = eigenvectors[currentPair[2], orbitalIndex]
				nextPol[1] = eigenvectors[nextPair[1],orbitalIndex]
				nextPol[2] = eigenvectors[nextPair[2],orbitalIndex]
				# Check the rotation matrix and its correct direction
				currentPolRot = [cos((siteIndex-1)*v) -sin((siteIndex-1)*v);sin((siteIndex-1)*v) cos((siteIndex-1)*v)] * currentPol
				nextPolRot = [cos((siteIndex)*v) -sin((siteIndex)*v);sin((siteIndex)*v) cos((siteIndex)*v)] * nextPol
				helicity = currentPolRot[1] * nextPolRot[2] - currentPolRot[2] * nextPolRot[1]
				write(file, @sprintf("%g %g %g %g\n", siteIndex, currentPolRot..., helicity))
			end
			# Last site is further rotated by xi
			siteIndex = length(polarisationIndexPairs)-1
			currentPair = polarisationIndexPairs[siteIndex]
			nextPair = polarisationIndexPairs[siteIndex+1]
			currentPol[1] = eigenvectors[currentPair[1], orbitalIndex]
			currentPol[2] = eigenvectors[currentPair[2], orbitalIndex]
			nextPol[1] = eigenvectors[nextPair[1],orbitalIndex]
			nextPol[2] = eigenvectors[nextPair[2],orbitalIndex]
			# Check the rotation matrix and its correct direction
			currentPolRot = [cos((siteIndex-1)*v+extraAngle) -sin((siteIndex-1)*v+extraAngle);sin((siteIndex-1)*v+extraAngle) cos((siteIndex-1)*v+extraAngle)] * currentPol
			nextPolRot = [cos((siteIndex)*v+extraAngle) -sin((siteIndex)*v+extraAngle);sin((siteIndex)*v+extraAngle) cos((siteIndex)*v+extraAngle)] * nextPol
			helicity = currentPolRot[1] * nextPolRot[2] - currentPolRot[2] * nextPolRot[1]
			write(file, @sprintf("%g %g %g %g\n", siteIndex, currentPolRot..., helicity))
			# Print out the last cite with the same helicity
			write(file, @sprintf("%g %g %g %g\n", length(polarisationIndexPairs), nextPolRot..., helicity))
			close(file)
		end
	end
end
