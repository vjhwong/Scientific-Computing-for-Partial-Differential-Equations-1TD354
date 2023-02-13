import numpy as np
import matplotlib.pyplot as plt

X = np.array(
    (
        [1.4, 1.4],
        [2.2, 2.2],
        [0.2, 0.8],
        [1, 2.8],
        [0.6, 0.2],
        [0.4, 2.2],
        [0.6, 1.4],
        [1.2, 1.8],
        [1.2, 0.6],
        [1.8, 0.6],
    )
)

y = np.array((0.1, 0.6, -0.7, -1.8, 0.3, -1.9, -0.9, -0.5, 0.7, 1.5))

fix, ax = plt.subplots()
ax.scatter(X[:, 0].flatten(), X[:, 1].flatten())
ax.scatter(1.5, 1.8, c="red")

for i, txt in enumerate(y):
    ax.annotate(txt, (X[:, 0][i], X[:, 1][i]))
ax.annotate((0.6 + 0.5) / 2, (1.5, 1.8))
plt.axhline(y=1.7, xmin=0.0, xmax=1.1 / 2.4)
plt.axhline(y=0.9, xmin=1.1 / 2.4, xmax=1)
plt.axhline(y=0.5, xmin=0.0, xmax=1.1 / 2.4)
plt.axhline(y=1.6, xmin=1.1 / 2.4, xmax=1)
plt.axvline(x=1.1)
plt.show()
