import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, epochs=10000, learning_rate=0.01):
    input_size = X.shape[1]
    hidden_size = 3
    output_size = 1

    # Initialize random weights and biases
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)

        # Calculate error
        error = y - predicted_output

        # Backward pass
        output_delta = error * sigmoid_derivative(predicted_output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

        # Print mean squared error for every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

trained_weights_input_hidden, trained_bias_hidden, trained_weights_hidden_output, trained_bias_output = \
    train_neural_network(X, y, epochs=10000, learning_rate=0.01)

# Test the trained model
hidden_layer_input = np.dot(X, trained_weights_input_hidden) + trained_bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, trained_weights_hidden_output) + trained_bias_output
predictions = sigmoid(output_layer_input)

print("Predictions:")
print(predictions)
