"""Logistic regression baseline for PlantVillage grayscale images."""
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from pathlib import Path

class GreyScale:
    def __init__(self) -> None:
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> None:
        project_root = Path(__file__).resolve().parents[2]

        grayscale_dir = (
            project_root
            / "data"
            / "raw"
            / "plantvillage"
            / "plantvillage dataset"
            / "grayscale"
        )

        X = []
        y = []

        image_size = (32,32)

        for class_folder in grayscale_dir.iterdir():
            if not class_folder.is_dir():
                continue

            label = class_folder.name

            for image_path in class_folder.iterdir():
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                image = Image.open(image_path).convert("L")
                image = image.resize(image_size)

                pixels = np.array(image, dtype=np.float32).flatten()

                pixels = pixels / 255.0

                X.append(pixels)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print("Loaded data.")
        print(f"Total images: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Number of classes: {len(np.unique(y))}")


class RegressionModel(GreyScale):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=1000) #classweight='balanced' 

    def train(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been loaded.")

        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate(self) -> None:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data has not been loaded.")

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    def save_model(self, path: Path) -> None:
        """Provide path with .joblib extension"""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")


class RegressionPCA(RegressionModel):
    def __init__(self, n_components) -> None:
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def train(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Error with loading the data.")
        
        print(f"Training PCA model with n_components={self.n_components}")

        X_train_pca = self.pca.fit_transform(self.X_train)
        self.model.fit(X_train_pca, self.y_train)
        print("PCA model is finished training.")

    def evaluate(self) -> None:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Error with loading the data.")

        X_test_pca = self.pca.transform(self.X_test)
        y_pred = self.model.predict(X_test_pca)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

   
    def scree_plot(self, max_components=300) -> None:
        if self.X_train is None:
            raise ValueError("Data has not been loaded.")

        max_components = min(max_components, self.X_train.shape[1])

        pca = PCA(n_components=max_components)
        pca.fit(self.X_train)

        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, max_components + 1),
            explained_variance,
            marker="o",
            linestyle="-"
        )
        plt.title("Scree Plot")
        plt.xlabel("Dimension")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    print("Linear Regression model","="*50)
    model1 = RegressionModel()
    

    model1.load_data()
    model1.train()
    model1.evaluate()
    model1.save_model(Path("models/logistic_regression.joblib"))

    print("Model 2","="*50)
    model2 = RegressionPCA(n_components=50)
    model2.load_data()
    model2.train()
    model2.evaluate()
    model2.scree_plot()
    model2.save_model(Path("models/logistic_regression_pca_50.joblib"))