# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset.

2.Convert the dataset into a DataFrame.

3.Add the target column to the DataFrame.

4.Separate the independent variables X and dependent variable Y.

5.Split the data into training and testing sets.

6.Create the SGD classifier model.

7.Train the model using the training data.

8.Predict the class labels for the test data.

9.Calculate the accuracy of the model.

10.Generate the confusion matrix.

11.Display the confusion matrix using a heatmap.

## Program:

Program to implement the prediction of iris species using SGD Classifier.

Developed by: SRIRAM ARUN S

RegisterNumber:  212225040429
```python

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt='d',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```
## Output:
<img width="891" height="407" alt="image" src="https://github.com/user-attachments/assets/d9fbc056-ead0-468e-a696-ba0c2873b7cf" />
<img width="696" height="523" alt="image" src="https://github.com/user-attachments/assets/02de078e-e10d-4f6e-bfe0-72f3c12aa553" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
