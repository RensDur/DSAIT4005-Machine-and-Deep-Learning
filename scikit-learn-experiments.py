import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression

X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LogisticRegression()

clf.fit(X_train, y_train)

# Compute accuracy on training-set

y_pred = clf.predict(X_train)

accuracy = np.mean(y_pred == y_train)
print('Accuracy on the training set:', accuracy)

# Compute accuracy on test-set

y_pred = clf.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print('Accuracy on the test set:', accuracy)

# Plot the decision boundary

x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))

Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()]) # equivalent to np.column_stack([xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.5, cmap='viridis')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision boundary')
plt.show()