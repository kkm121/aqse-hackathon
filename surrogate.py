# surrogate_model.py (improved with per-output isotonic calibration)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

class SurrogateAQSE:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.iso_list = []   # one isotonic regressor per output
        self.is_fitted = False

    def fit(self, X, Y, test_size=0.2):
        X = np.asarray(X)
        Y = np.asarray(Y)

        # If Y is 1D → reshape to (n_samples, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Split train/test
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )

        # Train RF model (handles multi-output automatically)
        self.model.fit(X_train, Y_train)

        # Reset isotonic regressors
        self.iso_list = []

        # For each output dimension, calibrate separately
        for j in range(Y.shape[1]):
            # Ensemble predictions for std estimation
            train_preds = np.stack(
                [tree.predict(X_train)[:, j] for tree in self.model.estimators_],
                axis=0
            )
            train_stds = train_preds.std(axis=0)

            # Validation residuals
            val_preds = self.model.predict(X_test)[:, j]
            residuals = np.abs(val_preds - Y_test[:, j])

            # Match shapes
            std_proxy = np.interp(
                np.linspace(0, 1, len(residuals)),
                np.linspace(0, 1, len(train_stds)),
                np.sort(train_stds)
            )

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(std_proxy, np.sort(residuals))
            self.iso_list.append(iso)

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("SurrogateAQSE not fitted yet!")

        X = np.asarray(X)
        mean_preds = self.model.predict(X)

        if mean_preds.ndim == 1:
            mean_preds = mean_preds.reshape(-1, 1)

        # Ensemble predictions for uncertainty
        all_preds = np.stack(
            [tree.predict(X) for tree in self.model.estimators_],
            axis=0
        )  # shape: (n_estimators, n_samples, n_outputs)

        raw_stds = all_preds.std(axis=0)  # (n_samples, n_outputs)

        # Calibrate each output
        calibrated_stds = np.zeros_like(raw_stds)
        for j, iso in enumerate(self.iso_list):
            calibrated_stds[:, j] = iso.predict(raw_stds[:, j])

        return mean_preds, calibrated_stds
