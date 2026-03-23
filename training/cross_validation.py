from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import compute_metrics

class CrossValidator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def run(self, X, y):
        skf = StratifiedKFold(
            n_splits=self.config["n_folds"],
            shuffle=True,
            random_state=self.config["seed"]
        )

        results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)

            metrics = compute_metrics(y_val, preds)
            results.append(metrics)

        return results
