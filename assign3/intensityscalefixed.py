import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import imageio
import math
import cv2
import matplotlib.pyplot as plt

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

def hybridizeImage(FrequencyImg, highSigmaCutoff, lowSigmaCutoff):
    lowPassImage = lowPass(FrequencyImg, lowSigmaCutoff)
    highPassImage = highPass(FrequencyImg, highSigmaCutoff)
    
    return lowPassImage, highPassImage 

# scale image's intensity to [0,1] with mean value of 0.5 for better visualization.
def intensityscale(raw_img):
    # scale an image's intensity from [min, max] to [0, 1]
    v_min, v_max = raw_img.min(), raw_img.max()
    scaled_im = (raw_img * 1.0 - v_min) / (v_max - v_min)
    # keep the mean to be 0.5

    meangray = np.mean(scaled_im)
    scaled_im = scaled_im - meangray + 0.5

    # clip to [0, 1]
    scaled_im = np.clip(scaled_im, 0, 1)

    return scaled_im

im = cv2.imread('einsteinandwho.png', cv2.IMREAD_GRAYSCALE) / 255
im_scaled = intensityscale(im)
img = imageio.imread("einsteinandwho.png", as_gray=True)

lowSigma = 5
highSigma = 25

# You can remove them relatively easy with a 2D Fourier Transform (FFT), 
# Apply a low pass 2D filter (you can keep the low frequency content of the 
# spectrum and force to zero the others) and then apply the 2D IFFT

hybridLow, hybridHigh = hybridizeImage(img, lowSigma, highSigma)

hybridHigh = img - hybridLow

hybridLow = intensityscale(hybridLow)
hybridHigh  = intensityscale(hybridHigh)
# plt.imshow(hybridLow)
# plt.imshow(hybridHigh)
imageio.imwrite("lowPASSEDimg.png", np.real(hybridLow))
imageio.imwrite("highPASSEDimg.png", np.real(hybridHigh))
plt.show()