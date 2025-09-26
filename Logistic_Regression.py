import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) 

def calculate_gradient(theta, X, y):
    m = y.size
    predictions = sigmoid(X @ theta)       # (m,)
    return (X.T @ (predictions - y)) / m   # (n,)

def gradient_descent(X, y, alpha, tol=1e-7, num_iter=100): 
    x_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(x_b.shape[1])

    for i in range(num_iter): 
        grad = calculate_gradient(theta, x_b, y)
        theta -= alpha * grad 
        if np.linalg.norm(grad) < tol: 
            break
    return theta 

def predict_proba(X, theta): 
    x_b = np.c_[np.ones((X.shape[0], 1)), X]   # fixed bias
    return sigmoid(x_b @ theta) 

def prediction(X, theta): 
    return (predict_proba(X, theta) >= 0.5).astype(int)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # small fix: use transform, not fit_transform

theta = gradient_descent(X_train_scaled, y_train, alpha=0.1)

y_pred_train = prediction(X_train_scaled, theta) 
y_pred_test = prediction(X_test_scaled, theta) 

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(train_acc)
print(test_acc)
