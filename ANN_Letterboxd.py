"""Mini-batch NumPy adaptation of ANN.py for Letterboxd star-rating regression."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data import load_and_clean_data, split_reviews


class LetterboxdNetwork:
    """Fully connected ReLU network with a single linear rating output."""

    def __init__(self, n_features, hidden_layers=(32, 16), random_state=42):
        sizes = (n_features, *hidden_layers, 1)
        if any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError("Layer sizes must be positive integers.")
        if not hidden_layers:
            raise ValueError("Provide at least one hidden layer.")

        self.rng = np.random.default_rng(random_state)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialization keeps ReLU activations from vanishing with depth.
            weight = self.rng.normal(0, np.sqrt(2 / fan_in), (fan_out, fan_in))
            self.weights.append(weight.astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    def _forward(self, X):
        activations = [X]
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Sparse TF-IDF multiplies directly into the small first hidden layer.
            output = activations[-1] @ weight.T + bias
            if layer < len(self.weights) - 1:
                output = np.maximum(output, 0)
            activations.append(output)
        return activations

    def predict(self, X):
        """Return ratings within the legal range, without rounding to half-stars."""
        return np.clip(self._forward(X)[-1].ravel(), 0.5, 5.0)

    def _gradients(self, X, y, l2):
        activations = self._forward(X)
        # Linear output + mean squared error: no saturating output sigmoid.
        delta = 2 * (activations[-1] - y[:, None]) / len(y)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            # This order also supports sparse inputs without densifying TF-IDF.
            weight_gradients[layer] = (
                activations[layer].T @ delta
            ).T + l2 * self.weights[layer]
            bias_gradients[layer] = delta.sum(axis=0, keepdims=True)
            if layer:
                delta = (delta @ self.weights[layer]) * (activations[layer] > 0)
        return weight_gradients, bias_gradients

    def _validate_data(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.weights[0].shape[1]:
            raise ValueError("Feature count must match the network input layer.")
        if y.ndim != 1 or len(y) == 0 or X.shape[0] != len(y):
            raise ValueError("Provide one numeric rating per nonempty feature row.")
        values = X.data if issparse(X) else X
        if not np.isfinite(values).all() or not np.isfinite(y).all():
            raise ValueError("Features and ratings must contain only finite numbers.")
        if np.any((y < 0.5) | (y > 5.0)):
            raise ValueError("Ratings must lie between 0.5 and 5.")
        return y

    def train(self, X_train, y_train, X_val, y_val, eta=0.003, epochs=40,
              batch_size=128, patience=6, l2=0.0001, verbose=True):
        """Train with Adam, plateau-based learning-rate decay, and best-weight restore."""
        if not np.isfinite(eta) or eta <= 0 or not np.isfinite(l2) or l2 < 0:
            raise ValueError("Learning rate must be positive and L2 nonnegative.")
        if any(not isinstance(value, int) or value <= 0
               for value in (epochs, batch_size, patience)):
            raise ValueError("Epochs, batch size, and patience must be positive integers.")
        y_train = self._validate_data(X_train, y_train)
        y_val = self._validate_data(X_val, y_val)

        # Start near the training mean instead of making ratings near zero.
        self.biases[-1].fill(y_train.mean())
        parameters = self.weights + self.biases
        first_moments = [np.zeros_like(p) for p in parameters]
        second_moments = [np.zeros_like(p) for p in parameters]
        best_parameters = [p.copy() for p in parameters]
        best_mae = np.inf
        plateau_mae = np.inf
        stale_epochs = 0
        step = 0
        learning_rate = eta
        self.history_ = []
        started = perf_counter()

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(y_train))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                dW, db = self._gradients(X_train[indices], y_train[indices], l2)
                step += 1
                # Adam rescales updates using running gradient means and variances.
                for p, gradient, m, v in zip(
                    parameters, dW + db, first_moments, second_moments
                ):
                    m *= 0.9
                    m += 0.1 * gradient
                    v *= 0.999
                    v += 0.001 * gradient**2
                    p -= learning_rate * (m / (1 - 0.9**step)) / (
                        np.sqrt(v / (1 - 0.999**step)) + 1e-8
                    )

            # Evaluate once per epoch, not on the full dataset after every sample.
            predictions = self.predict(X_val)
            val_mae = float(np.mean(np.abs(predictions - y_val)))
            if not np.isfinite(val_mae) or not all(
                np.isfinite(p).all() for p in parameters
            ):
                raise FloatingPointError("Training diverged; try a smaller learning rate.")
            self.history_.append({
                "epoch": epoch, "learning_rate": learning_rate, "val_mae": val_mae
            })
            if verbose:
                print(f"Epoch {epoch:2d} | lr={learning_rate:.6f} | "
                      f"validation MAE={val_mae:.4f}")

            if val_mae < best_mae:
                best_mae = val_mae
                self.best_epoch_ = epoch
                best_parameters = [p.copy() for p in parameters]

            # Save even small gains, but require a useful gain to reset patience.
            if val_mae < plateau_mae - 0.001:
                plateau_mae = val_mae
                stale_epochs = 0
            else:
                stale_epochs += 1
                # Halve the rate after every two epochs without useful improvement.
                if stale_epochs % 2 == 0:
                    learning_rate = max(eta / 100, learning_rate * 0.5)
                if stale_epochs >= patience:
                    break

        for parameter, best in zip(parameters, best_parameters):
            parameter[...] = best
        self.training_seconds_ = perf_counter() - started
        return self.history_


def plot_performance(network, y_test, predictions, baseline,
                     output_path=Path(__file__).with_name("letterboxd_performance.png")):
    """Save four diagnostics using the recorded history and final test predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    fig.suptitle(
        f"Letterboxd rating prediction | training: {network.training_seconds_:.2f}s",
        fontsize=15,
    )

    epochs = [entry["epoch"] for entry in network.history_]
    val_mae = [entry["val_mae"] for entry in network.history_]
    rates = [entry["learning_rate"] for entry in network.history_]
    learning = axes[0, 0]
    learning.plot(epochs, val_mae, "o-", label="Validation MAE")
    learning.axvline(network.best_epoch_, color="tab:green", linestyle=":",
                     label=f"Best epoch ({network.best_epoch_})")
    learning.set(title="Learning curve and learning-rate schedule",
                 xlabel="Epoch", ylabel="MAE (stars)", xticks=epochs)
    learning.grid(alpha=0.2)
    rate_axis = learning.twinx()
    rate_axis.step(epochs, rates, where="post", color="tab:orange",
                   linestyle="--", label="Learning rate")
    rate_axis.set(ylabel="Learning rate", yscale="log")
    lines, labels = learning.get_legend_handles_labels()
    rate_lines, rate_labels = rate_axis.get_legend_handles_labels()
    learning.legend(lines + rate_lines, labels + rate_labels, fontsize=8)

    ratings = np.arange(0.5, 5.5, 0.5)
    actual = axes[0, 1]
    actual.scatter(y_test, predictions, alpha=0.2, s=14, edgecolors="none")
    actual.plot([0.5, 5], [0.5, 5], "k--", label="Perfect prediction")
    actual.set(title="Actual vs. predicted test ratings",
               xlabel="Actual rating (stars)", ylabel="Predicted rating (stars)",
               xlim=(0.3, 5.2), ylim=(0.3, 5.2), xticks=ratings, yticks=ratings)
    actual.set_aspect("equal", adjustable="box")
    actual.legend()
    actual.grid(alpha=0.2)

    observed_ratings, counts = np.unique(y_test, return_counts=True)
    rating_mae = [
        np.mean(np.abs(predictions[y_test == rating] - rating))
        for rating in observed_ratings
    ]
    by_rating = axes[1, 0]
    by_rating.bar(observed_ratings, rating_mae, width=0.35)
    for rating, error, count in zip(observed_ratings, rating_mae, counts):
        by_rating.annotate(f"n={count}", (rating, error), xytext=(0, 5),
                           textcoords="offset points", ha="center", fontsize=8)
    by_rating.set(title="Test MAE by actual rating (labels show sample counts)",
                  xlabel="Actual rating (stars)", ylabel="MAE (stars)",
                  xticks=ratings, ylim=(0, max(0.1, max(rating_mae) * 1.25)))
    by_rating.grid(axis="y", alpha=0.2)

    comparison = axes[1, 1]
    positions = np.arange(2)
    for offset, label, estimates in (
        (-0.18, "Network", predictions),
        (0.18, "Training-mean baseline", baseline),
    ):
        errors = estimates - y_test
        scores = [np.mean(np.abs(errors)), np.sqrt(np.mean(errors**2))]
        bars = comparison.bar(positions + offset, scores, width=0.36, label=label)
        comparison.bar_label(bars, fmt="%.3f", padding=3)
    comparison.set(title="Held-out test error (lower is better)",
                   ylabel="Error (stars)", xticks=positions, xticklabels=["MAE", "RMSE"])
    comparison.margins(y=0.2)
    comparison.legend()
    comparison.grid(axis="y", alpha=0.2)

    fig.savefig(output_path, dpi=160)
    print(f"Performance plots saved to: {output_path}")
    return fig


def train_letterboxd():
    df = load_and_clean_data()
    train_val, test = split_reviews(df)
    train, validation = split_reviews(train_val, random_state=43)

    # Neither validation nor test text may influence the learned vocabulary.
    vectorizer = TfidfVectorizer(max_features=1000, dtype=np.float32)
    X_train = vectorizer.fit_transform(train["Review"])
    X_val = vectorizer.transform(validation["Review"])
    y_train = train["rating_num"].to_numpy(dtype=np.float32)
    y_val = validation["rating_num"].to_numpy(dtype=np.float32)

    network = LetterboxdNetwork(X_train.shape[1])
    print(f"\nArchitecture: {X_train.shape[1]} -> 32 -> 16 -> 1")
    print(f"Reviews: train={len(train)}, validation={len(validation)}, test={len(test)}")
    network.train(X_train, y_train, X_val, y_val)
    print(f"Training: {network.training_seconds_:.2f}s; "
          f"best epoch: {network.best_epoch_}")

    # The held-out test set is evaluated only after architecture/training decisions.
    X_test = vectorizer.transform(test["Review"])
    y_test = test["rating_num"].to_numpy(dtype=np.float32)
    predictions = network.predict(X_test)
    baseline = np.full_like(y_test, y_train.mean())
    for label, ratings in (("Network", predictions), ("Training-mean baseline", baseline)):
        mae = np.mean(np.abs(ratings - y_test))
        rmse = np.sqrt(np.mean((ratings - y_test)**2))
        print(f"{label}: test MAE={mae:.4f}, RMSE={rmse:.4f} stars")
    plot_performance(network, y_test, predictions, baseline)
    return network, vectorizer


if __name__ == "__main__":
    network, vectorizer = train_letterboxd()
    plt.show()