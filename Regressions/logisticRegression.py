import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

# X represents the size of a tumor in centimeters.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88])
# Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
X = X.reshape(-1, 1)
# y represents whether the tumor is cancerous (0 for "No", 1 for "Yes").
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

# predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(np.array([3.46]).reshape(-1, 1))
print(predicted)

log_odds = logr.coef_
print(log_odds)
print(logr.intercept_)
odds = np.exp(log_odds)
print(odds)


def prob(x):
    w = logr.coef_
    b = logr.intercept_
    z = w * x + b
    return 1 / (1 + np.exp(-z))


def prob1(x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


def odd1(x):
    return prob(x) / (1 - prob(x))


def odd2(x):
    w = logr.coef_
    b = logr.intercept_
    z = w * x + b
    return np.exp(z)


X = X.reshape(1, -1)
arr = np.linspace(X.min(), X.max(), 50)

model = prob1(arr)
f_mod = prob(arr)
odds1 = odd1(arr)
odds2 = odd2(arr)

print(model)
print(f_mod)
print(odds1)
print(odds2)

plt.subplot(1, 2, 1)
plt.scatter(X, y)
plt.plot(arr, model[0])
plt.plot(arr, f_mod[0])
plt.xlabel("Size of tumor")
plt.ylabel("Probability of being malignant")
plt.title('Probability')

plt.subplot(1, 2, 2)
plt.plot(arr, odds1[0])
plt.plot(arr, odds2[0])

plt.show()
