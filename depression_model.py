import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example synthetic dataset
data = {
    "sleep_quality": [2, 3, 1, 4, 5, 2, 1, 3, 4, 2],
    "stress_level": [5, 4, 5, 2, 1, 4, 5, 3, 2, 4],
    "mood_score": [2, 3, 1, 4, 5, 2, 1, 3, 4, 2],
    "depression_risk": [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[["sleep_quality", "stress_level", "mood_score"]]
y = df["depression_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)
import matplotlib.pyplot as plt

plt.bar(["Accuracy"], [accuracy])
plt.title("Model Performance")
plt.ylabel("Score")
plt.show()
