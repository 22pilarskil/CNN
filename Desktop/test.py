X = [[0], [1], [2], [3]]
y = [0, 2, 4, 6]
import numpy as np 
from sklearn import neighbors
neigh = neighbors.KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 

print(neigh.predict([[3]]))

print(neigh.score(y, X))
