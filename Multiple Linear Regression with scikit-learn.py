import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34],[60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
X,Y = np.array(x), np.array(y)
print(X)
print(Y)

model=LinearRegression()
model.fit(X,Y)
print('intercept:', model.intercept_)

print('slope:', model.coef_)

# plt.title("X vs Y")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.scatter(X,Y)
# plt.plot(Y, (model.coef_*X)+(model.intercept_))
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

x_new = np.arange(10).reshape((-1, 2))

print("New values of x")
print(x_new)

y_new = model.predict(x_new)

print("New value of y")
print(y_new)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)