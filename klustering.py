from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
features, true_labels = make_blobs(n_samples=200, centers=3, cluster_std=2.75, random_state=42)
print('Features \n\n', features, '\n\n\n')
print('true_labels\n\n', true_labels, '\n\n\n')


plt.scatter(features[:,0], features[:,1])
plt.show()