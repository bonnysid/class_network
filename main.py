import numpy as np
import matplotlib.pyplot as plt

# Constants
INPUT_DIM = 2
OUTPUT_DIM = 3
H_DIM = 5
LEARNING_RATE = 0.001

COUNT_EXAMPLES = 500
NUM_EPOCHS = 100
CLASS_NAMES = ['Cat', 'Dog', 'Mouse']

W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.randn(1, H_DIM)
W2 = np.random.randn(H_DIM, OUTPUT_DIM)
b2 = np.random.randn(1, OUTPUT_DIM)

def relu(t):
    return np.maximum(t, 0)


def relu_deriv(t):
    return (t >= 0).astype(float)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


# Initialize train matrix
cat_matrix = np.random.randn(COUNT_EXAMPLES, INPUT_DIM) + np.array([0, -3])
dog_matrix = np.random.randn(COUNT_EXAMPLES, INPUT_DIM) + np.array([3, 3])
mouse_matrix = np.random.randn(COUNT_EXAMPLES, INPUT_DIM) + np.array([-3, 3])

labels = np.array([0]*COUNT_EXAMPLES + [1]*COUNT_EXAMPLES + [2]*COUNT_EXAMPLES)
training_inputs = np.vstack([cat_matrix, dog_matrix, mouse_matrix])
training_outputs = np.array([0] * COUNT_EXAMPLES + [1] * COUNT_EXAMPLES + [2] * COUNT_EXAMPLES)

# Show classification diagram
plt.figure(figsize=(10,7))
plt.scatter(training_inputs[:,0], training_inputs[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
plt.show()

dataset = [(training_inputs[i][None, ...], training_outputs[i]) for i in range(len(training_inputs))]

loss_arr = []

for ep in range(NUM_EPOCHS):
    for i in range(len(dataset)):
        x, y = dataset[i]

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)

        # Backward
        y_full = to_full(y, OUTPUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        # Update
        W1 = W1 - LEARNING_RATE * dE_dW1
        b1 = b1 - LEARNING_RATE * dE_db1
        W2 = W2 - LEARNING_RATE * dE_dW2
        b2 = b2 - LEARNING_RATE * dE_db2

        loss_arr.append(E)


def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy():
    correct = 0
    for i in range(len(training_inputs)):
        z = predict(training_inputs[i])
        y_pred = np.argmax(z)
        if y_pred == training_outputs[i]:
            correct += 1
    acc = correct / len(training_inputs)
    return acc


accuracy = calc_accuracy()
print("Accuracy:", accuracy)

# plt.plot(loss_arr)
# plt.show()

print("Test 1:", CLASS_NAMES[np.argmax(predict(dataset[0][0]))], CLASS_NAMES[dataset[0][1]])
print("Test 2:", CLASS_NAMES[np.argmax(predict(dataset[501][0]))], CLASS_NAMES[dataset[501][1]])
print("Test 3:", CLASS_NAMES[np.argmax(predict(dataset[1001][0]))], CLASS_NAMES[dataset[1001][1]])

print("Test 4 (not from dataset):", CLASS_NAMES[np.argmax(predict(np.array([[0, -1]])))], CLASS_NAMES[0])
print("Test 5 (not from dataset):", CLASS_NAMES[np.argmax(predict(np.array([[3, 2]])))], CLASS_NAMES[1])
print("Test 6 (not from dataset):", CLASS_NAMES[np.argmax(predict(np.array([[-3, 2]])))], CLASS_NAMES[2])

print("Test 7 (out of area):", CLASS_NAMES[np.argmax(predict(np.array([[10, 10]])))], CLASS_NAMES[1])
print("Test 8 (out of area):", CLASS_NAMES[np.argmax(predict(np.array([[-10, 10]])))], CLASS_NAMES[2])
print("Test 9 (out of area):", CLASS_NAMES[np.argmax(predict(np.array([[0, -10]])))], CLASS_NAMES[0])