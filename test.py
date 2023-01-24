from birch import*
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import random
from sklearn.cluster import Birch

random.seed(1)
x, _ = make_blobs(n_samples=400, centers=5, cluster_std=1.2)
bclust = Birch_implementation(branching_factor=200, n_clusters=None, threshold=1,compute_labels=True).fit(x)
bclust2=Birch(branching_factor=200, threshold = 1).fit(x)

plt.figure(figsize=(15, 10))
labels = bclust.predict(x)
labels2 = bclust2.predict(x)
plt.subplot(131) # dane
plt.scatter(x[:,0], x[:,1])
plt.subplot(132) # birch z biblioteki
plt.scatter(x[:,0], x[:,1], c=labels)
plt.subplot(133) # implementacja
plt.scatter(x[:,0], x[:,1], c=labels2)

plt.show()
