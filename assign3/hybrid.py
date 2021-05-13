import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import imageio
import math
import cv2

def makeGaussianFilter(numberOfRows, numberOfCols, sigma, highPass=True):
	centerY = 0
	centerX = 0
	# Gets the center of the image to Gaussian Filter
	if(numberOfRows % 2 == 1):
		centerY = int(numberOfRows / 2) + 1
	else:
		centerY = int(numberOfRows / 2)

	if(numberOfRows % 2 == 1):
		centerX = int(numberOfRows / 2) + 1
	else:
		centerX = int(numberOfRows / 2)

	def gaussian(i,j):
		# Gaussian Filter function
		coefficient = math.exp(-1.0 * ((i - centerY)**2 + (j - centerX)**2) / (2 * sigma**2))

		# This handles (A - blur(A)) for the low pass image
		if(highPass):
			return 1 - coefficient
		else:
			return coefficient

	# List comprehension to create 2D array of gaussian coefficients at i,j
	return np.array([[gaussian(i,j) for j in range(numberOfCols)] for i in range(numberOfRows)])

def filterDFourierTransform(imageMatrix, filteredMatrix):
   	shiftedDFT = fftshift(fft2(imageMatrix))
   	filteredDFT = shiftedDFT * filteredMatrix
   	return ifft2(ifftshift(filteredDFT))

def lowPass(imageMatrix, sigma):
   	n, m = imageMatrix.shape
   	gaussFiltered = makeGaussianFilter(n, m, sigma, highPass = False)
   	return filterDFourierTransform(imageMatrix, gaussFiltered)

def highPass(imageMatrix, sigma):
   	n, m = imageMatrix.shape
   	gaussFiltered = makeGaussianFilter(n, m, sigma, highPass = True)
   	return filterDFourierTransform(imageMatrix, gaussFiltered)

def hybridizeImage(highFrequencyImg, lowFrequencyImg, highSigmaCutoff, lowSigmaCutoff):
	lowPassImage = lowPass(lowFrequencyImg, lowSigmaCutoff)
	highPassImage = highPass(highFrequencyImg, highSigmaCutoff)
	
	return lowPassImage + highPassImage


# import and resize images to be used
img1 = imageio.imread("paulrudd.jpg", as_gray=True)
img1 = cv2.resize(img1, dsize=(900, 900), interpolation = cv2.INTER_CUBIC)
img2 = imageio.imread("markruffalo.jpg", as_gray=True)
img2 = cv2.resize(img2, dsize=(900, 900), interpolation = cv2.INTER_CUBIC)

# Start making the hybrid image

sigmaStart = 35
for i in range(5, sigmaStart, 5):
	lowSigma = i
	for j in range(sigmaStart, 65, 5):
		highSigma = j
		hybrid = hybridizeImage(img1, img2, highSigma, lowSigma)
		filename = "rudd-ruffaloHigh%sLow%s.png" % (highSigma, lowSigma)
		imageio.imwrite(filename, np.real(hybrid))

