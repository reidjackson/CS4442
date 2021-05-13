import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.decomposition import PCA

imagesRaw = np.loadtxt('faces.dat')

# 3a
# hundredthImage = imagesRaw[99,:]
# hundredthImage = hundredthImage.reshape((64,64)).T

# plt.imshow(hundredthImage)
# plt.show()

# 3b
# for i in range (0, len(imagesRaw)):
# 	imageMean = np.mean(imagesRaw[i,:])
# 	imagesRaw[i,:] -= imageMean


# hundredthImage = imagesRaw[99,:]
# hundredthImage = hundredthImage.reshape((64,64)).T


# plt.imshow(hundredthImage)
# plt.show()

# 3c
# for i in range (0, len(imagesRaw)):
# 	imageMean = np.mean(imagesRaw[i,:])
# 	imagesRaw[i,:] -= imageMean

# pca = PCA()
# pca.fit(imagesRaw)
# # EIGENVALUES pca.explained_variance_
# eigenValues = (pca.explained_variance_)

# plt.plot(eigenValues, 'bo')
# plt.show()

# 3e

# for i in range (0, len(imagesRaw)):
# 	imageMean = np.mean(imagesRaw[i,:])
# 	imagesRaw[i,:] -= imageMean

# pca = PCA()
# pca.fit(imagesRaw)
# # EIGENVALUES pca.explained_variance_
# eigenValues = (pca.explained_variance_)

# dropThreshold = sum(eigenValues) * 0.977
# index = 0
# eigenSum = 0

# while(eigenSum < dropThreshold):
# 	eigenSum += eigenValues[index]
# 	index += 1

# print(index)

# 3f
# for i in range (0, len(imagesRaw)):
# 	imageMean = np.mean(imagesRaw[i,:])
# 	imagesRaw[i,:] -= imageMean

# pca = PCA()
# pca.fit(imagesRaw)

# for i in range (0, 5):
# 	pcaImage = pca.components_[i]
# 	pcaImage = np.asarray(pcaImage).reshape((64,64)).T
# 	plt.imshow(pcaImage)
# 	plt.show()

# 3g
for i in range (0, len(imagesRaw)):
	imageMean = np.mean(imagesRaw[i,:])
	imagesRaw[i,:] -= imageMean

pca = PCA()
pca.fit(imagesRaw)

hundredthImage = imagesRaw[99,:]
hundredthImage = hundredthImage.reshape((64,64)).T
principalComps = [10, 100, 200, 399]
reconstructed = np.zeros((64,64))

for i in principalComps:
	for j in range (0, i):
		currentVector = pca.components_[j]
		currentVector = np.asarray(currentVector).reshape((64,64)).T
		reconstructed = reconstructed + ((currentVector @ currentVector.T) @ hundredthImage)
	plt.imshow(reconstructed)
	plt.show()
		

