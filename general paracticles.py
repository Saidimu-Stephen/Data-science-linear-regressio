from sklearn.linear_model import  LinearRegression
from sklearn import  metrics

import matplotlib.pyplot as plt
import numpy as np
model=LinearRegression(fit_intercept=True)
print(LinearRegression())
rng = np.random.RandomState(42)
x = 10 * rng.rand(40)
y = 2 * x - 1 + rng.randn(40)
print (x)
print (y)
plt.scatter(x, y)
# plt.show()

X = x[:, np.newaxis]
print(X.shape)

model.fit(X,y)
print("coefficient", model.coef_)
print("Modele intercept", model.intercept_)


plt.title("X vs Y")
plt.scatter(x, y);
plt.plot(x, (model.coef_*x)+(model.intercept_),color='red')
plt.xlabel("X")
plt.ylabel("Y")
# plt.show()

xfit = np.linspace(-1, 11, 40)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# Checking the model performance
r_sq = model.score(X, y)
print('coefficient of determination:', r_sq)

#Cost function
print("Mean squared errror:", metrics.mean_squared_error(y, (model.coef_*x)+(model.intercept_)))