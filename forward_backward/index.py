import numpy as np
import matplotlib.pyplot as plt


# ----- Data Generation -----
def generate_data(num_samples=100, num_features=2):
    # num py set seed, so data will be same?
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


# ----- Neural Network Implementation -----
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim=1, learning_rate=0.01):
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.lr = learning_rate
        self.loss_history = []

    def sigmoid(self, Z):
        # Clip Z to prevent overflow
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def sigmoid_deriv(self, A):
        # A is the output of sigmoid(Z)
        # Derivative of sigmoid is: sigmoid(z) * (1 - sigmoid(z))
        return A * (1 - A)

    def forward(self, X):
        """
        Forward pass:
        1. z1 = X * W1 + b1
        2. a1 = Sigmoid(z1)
        3. z2 = a1 * W2 + b2
        4. a2 = Sigmoid(z2)  (for binary classification)

        Returns:
        a2 (final output) and cache containing all intermediate variables
        """
        # Layer 1: Input -> Hidden
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)

        # Layer 2: Hidden -> Output
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        # Store intermediate values for backward pass
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

        return a2, cache

    def backward(self, cache, y_true):
        """
        Backward pass:
        1. Compute loss = -(1/N)*sum(y*log(a2) + (1-y)*log(1-a2))
        2. Compute gradients w.r.t W2, b2, W1, b1
        """
        # Extract cached values
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        a2 = cache["a2"]

        N = X.shape[0]
        y_true = y_true.reshape(-1, 1)  # Ensure proper shape

        # Compute binary cross-entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        a2_clipped = np.clip(a2, epsilon, 1 - epsilon)
        loss = -(1 / N) * np.sum(
            y_true * np.log(a2_clipped) + (1 - y_true) * np.log(1 - a2_clipped)
        )

        # Backward propagation
        # Output layer gradients
        dz2 = a2 - y_true  # Derivative of loss w.r.t z2
        dW2 = (1 / N) * np.dot(a1.T, dz2)
        db2 = (1 / N) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)  # Derivative of loss w.r.t a1
        dz1 = da1 * self.sigmoid_deriv(a1)  # Derivative of loss w.r.t z1
        dW1 = (1 / N) * np.dot(X.T, dz1)
        db1 = (1 / N) * np.sum(dz1, axis=0, keepdims=True)

        # Update parameters using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            a2, cache = self.forward(X)

            # Backward pass
            loss = self.backward(cache, y)
            self.loss_history.append(loss)

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """Make predictions on new data"""
        a2, _ = self.forward(X)
        return (a2 > 0.5).astype(int).flatten()


# ----- Visualization Functions -----
def plot_decision_boundary(model, X, y):
    """Plot the decision boundary for 2D data"""
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return

    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = model.forward(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 4))

    # Plot decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.colorbar(label="Output Probability")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="black")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.colorbar(scatter, label="True Class")

    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ----- Main Execution -----
if __name__ == "__main__":
    # Generate data
    print("Generating synthetic binary classification data...")
    X, y = generate_data(num_samples=200, num_features=2)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Initialize and train model
    print("\nInitializing neural network...")
    model = SimpleNN(
        input_dim=X.shape[1], hidden_dim=8, output_dim=1, learning_rate=0.1
    )

    print("Training neural network...")
    model.train(X, y, epochs=1000)

    # Test final predictions
    print("\nEvaluating model...")
    preds, _ = model.forward(X)
    predicted_classes = (preds > 0.5).astype(int).flatten()
    accuracy = np.mean(predicted_classes == y)
    print(f"Final Training Accuracy: {accuracy:.2f}")

    # Print final parameters (optional)
    print("\nFinal model parameters:")
    print(f"W1 shape: {model.W1.shape}, W2 shape: {model.W2.shape}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")

    # Visualize results
    print("\nPlotting results...")
    plot_decision_boundary(model, X, y)

    # Test on some sample points
    print("\nTesting predictions on sample points:")
    test_points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    test_preds, _ = model.forward(test_points)
    test_classes = (test_preds > 0.5).astype(int).flatten()

    for i, (point, prob, pred_class) in enumerate(
        zip(test_points, test_preds.flatten(), test_classes)
    ):
        print(f"Point {point}: Probability={prob:.3f}, Predicted Class={pred_class}")
