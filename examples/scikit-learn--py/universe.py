# Imports for analyses
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from multiversum import Universe

universe = Universe(
    settings={
        # Configuration of this universe
        # (automatically overriden when running the multiverse analysis)
        "dimensions": {
            "scaler": "StandardScaler",
            "feature_selector": "SelectKBest_5",
            "model": "RandomForest",
        }
    }
)
# Get the parsed universe settings
dimensions = universe.dimensions

# Load data
data = load_wine()

X = data["data"]
y = data["target"]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=universe.seed
)

# Select the scaler based on the configuration
if dimensions["scaler"] == "StandardScaler":
    scaler = StandardScaler()
elif dimensions["scaler"] == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    scaler = None  # No scaling

# Select the feature selector based on the configuration
if dimensions["feature_selector"] == "SelectKBest_5":
    selector = SelectKBest(f_classif, k=5)
elif dimensions["feature_selector"] == "SelectKBest_10":
    selector = SelectKBest(f_classif, k=10)
else:
    selector = None  # No feature selection

# Select the model based on the configuration
if dimensions["model"] == "LogisticRegression":
    model = LogisticRegression(max_iter=1000)
elif dimensions["model"] == "DecisionTree":
    model = DecisionTreeClassifier()
else:
    model = RandomForestClassifier()

# Build the pipeline
steps = []
if scaler:
    steps.append(("scaler", scaler))
if selector:
    steps.append(("selector", selector))
steps.append(("classifier", model))

pipeline = Pipeline(steps)

# Fit the pipeline on the full training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Compute common classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Save the final metrics
final_data = pd.DataFrame(
    {
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
    }
)
universe.save_data(final_data)
