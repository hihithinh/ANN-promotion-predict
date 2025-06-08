# %% [markdown]
# # Predicting Employee Promotion with ANN from Scratch

# %% [markdown]
# ## 1. Import Libraries

# %% [code]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTENC

# %% [markdown]
# ## 2. Load Data

# %% [code]
# Load the training data
try:
    train_df = pd.read_csv('data/train.csv')
    print("Training data loaded successfully.")
    print("Shape of training data:", train_df.shape)
    print("First 5 rows:")
    print(train_df.head())
except FileNotFoundError:
    print("Error: 'data/train.csv' not found. Make sure the file is in the correct directory.")
    # Define train_df as None or an empty DataFrame to avoid NameError in subsequent cells if file not found
    train_df = None
except Exception as e:
    print(f"An error occurred: {e}")
    train_df = None

# %% [markdown]
# ## 3. Data Preprocessing

# %% [code]
if train_df is not None:
    print("\nData Info:")
    train_df.info()
    print("\nMissing values per column:")
    print(train_df.isnull().sum())

    # Drop employee_id and separate features (X) and target (y)
    X = train_df.drop(['employee_id', 'is_promoted'], axis=1)
    y = train_df['is_promoted'].values  # .values to get a NumPy array

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    print(f"\nCategorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get categorical feature indices for SMOTENC (before preprocessing)
    categorical_indices = []
    current_idx = 0

    # Store the original column order for reference
    original_features = X.columns.tolist()

    # Track the indices of categorical features after one-hot encoding
    for feature in original_features:
        if feature in categorical_features:
            # For categorical features, we'll need to know how many binary columns they'll become
            # We'll handle this after preprocessing
            categorical_indices.append(current_idx)
        current_idx += 1

    print(f"Categorical feature indices (before preprocessing): {categorical_indices}")

    # Apply the preprocessing pipeline
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Get the names of the output features after transformation
    output_feature_names = []

    # Extract numerical feature names (these stay the same)
    output_feature_names.extend(numerical_features)

    # Extract categorical feature names after one-hot encoding
    categorical_output_indices = []
    current_idx = len(numerical_features)

    # Get the one-hot encoder from the pipeline
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']

    # For each categorical feature, get the encoded feature names
    for i, cat_feature in enumerate(categorical_features):
        n_categories = len(ohe.categories_[i])
        # Mark all the one-hot encoded columns as categorical
        categorical_output_indices.extend(list(range(current_idx, current_idx + n_categories)))
        current_idx += n_categories

    print(f"Categorical feature indices (after preprocessing): {categorical_output_indices}")

    print(f"\nShape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val_processed: {X_val_processed.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Apply SMOTENC to the training data only
    print(f"\nOriginal y_train distribution: {np.bincount(y_train)}")

    # Apply SMOTENC with categorical_features parameter
    smotenc = SMOTENC(categorical_features=categorical_output_indices, random_state=42)
    X_train_smote, y_train_smote = smotenc.fit_resample(X_train_processed, y_train)
    print(f"Resampled y_train distribution after SMOTENC: {np.bincount(y_train_smote)}")
    print(f"Shape of X_train after SMOTENC: {X_train_smote.shape}")
    print(f"Shape of y_train after SMOTENC: {y_train_smote.shape}")

    # Transpose data for ANN input
    # Training data uses SMOTENC resampled data
    X_train_t = X_train_smote.T
    y_train_t = y_train_smote.reshape(1, -1)  # Ensure y_train_t is (1, m_train_smote)

    # Validation data remains unchanged (original distribution)
    X_val_t = X_val_processed.T
    y_val_t = y_val.reshape(1, -1)  # Ensure y_val_t is (1, m_val)

    print(f"\nShape of X_train_t (for ANN): {X_train_t.shape}")
    print(f"Shape of y_train_t (for ANN): {y_train_t.shape}")
    print(f"Shape of X_val_t (for ANN): {X_val_t.shape}")
    print(f"Shape of y_val_t (for ANN): {y_val_t.shape}")

else:
    print("train_df was not loaded. Skipping preprocessing and SMOTE.")
    # Define processed variables as None so subsequent cells can check for them
    X_train_processed, X_val_processed, y_train, y_val = None, None, None, None
    X_train_t, y_train_t, X_val_t, y_val_t = None, None, None, None  # Ensure these are defined


# %% [markdown]
# ## 4. ANN Implementation from Scratch

# %% [code]
def sigmoid(Z):
    """Implements the sigmoid activation function."""
    A = 1 / (1 + np.exp(-Z))
    cache = Z  # Store Z for backpropagation
    return A, cache


def relu(Z):
    """Implements the ReLU activation function."""
    A = np.maximum(0, Z)
    cache = Z  # Store Z for backpropagation
    return A, cache


def sigmoid_backward(dA, cache):
    """Implements the backward propagation for a single SIGMOID unit."""
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """Implements the backward propagation for a single RELU unit."""
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0  # When z <= 0, you should set dz to 0 as well.
    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(42)  # for reproducibility
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # He initialization for layers with ReLU, Xavier/Glorot for others if needed
        # For simplicity, using He for all hidden layers, Xavier for output if it were not sigmoid
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 # Small random values
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
            2. / layer_dims[l - 1])  # He initialization
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A_prev", "W", "b"; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, activations_list):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    activations_list -- list of activation functions for each layer (e.g. ["relu", "relu", "sigmoid"])

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing every cache of linear_activation_forward()
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        activation = activations_list[l - 1]  # activation for current layer l
        A, cache = linear_activation_forward(A_prev, W, b, activation)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID for the last layer
    W_L = parameters['W' + str(L)]
    b_L = parameters['b' + str(L)]
    activation_L = activations_list[L - 1]  # activation for output layer L
    AL, cache = linear_activation_forward(A, W_L, b_L, activation_L)
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function (binary cross-entropy).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]  # number of examples

    # Compute loss from AL and Y.
    cost = (-1 / m) * np.sum(
        Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))  # Added 1e-8 for numerical stability

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, activations_list):
    """
    Implement the backward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    activations_list -- list of activation functions used for each layer

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    # Derivative of the cost with respect to AL. For binary cross-entropy: -(Y/AL - (1-Y)/(1-AL))
    dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))  # Added 1e-8 for numerical stability

    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activations_list[L - 1])
    grads["dA" + str(L - 1)] = dA_prev_temp  # This is dA[L-1]
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        # dA_prev_temp is dA[l] for the current iteration, which was dA[l+1] in the previous one.
        # activations_list[l] refers to the activation of layer l+1 (1-indexed layer number)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activations_list[l])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def nn_model(X_train, Y_train, X_val, Y_val, layers_dims, activations_list, learning_rate=0.0075, num_iterations=3000,
             print_cost=False, print_cost_every=100):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X_train -- training set, numpy array of shape (input_size, number of examples)
    Y_train -- training labels, numpy array of shape (1, number of examples)
    X_val -- validation set
    Y_val -- validation labels
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    activations_list -- list of activation functions for each layer, e.g. ["relu", "relu", "sigmoid"]
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every print_cost_every iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs_train -- list of training costs recorded during training
    costs_val -- list of validation costs recorded during training
    """
    np.random.seed(1)
    costs_train = []  # keep track of training cost
    costs_val = []  # keep track of validation cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL_train, caches_train = L_model_forward(X_train, parameters, activations_list)

        # Compute training cost.
        cost_train = compute_cost(AL_train, Y_train)

        # Backward propagation.
        grads = L_model_backward(AL_train, Y_train, caches_train, activations_list)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Compute validation cost for plotting and monitoring
        AL_val, _ = L_model_forward(X_val, parameters, activations_list)
        cost_val = compute_cost(AL_val, Y_val)

        # Print the cost every print_cost_every training example
        if print_cost and i % print_cost_every == 0:
            print(f"Train Cost after iteration {i}: {cost_train:.4f}, Val Cost: {cost_val:.4f}")
        if i % print_cost_every == 0:
            costs_train.append(cost_train)
            costs_val.append(cost_val)

    return parameters, costs_train, costs_val


def predict(X, y, parameters, activations_list):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    y -- true labels (for accuracy calculation)
    parameters -- parameters of the trained model
    activations_list -- list of activation functions used in the model

    Returns:
    p -- predictions for the given dataset X (0 or 1)
    accuracy -- accuracy of the model
    """
    m = X.shape[1]  # number of examples
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters, activations_list)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    accuracy = np.sum((p == y) / m)
    print(f"Accuracy: {accuracy:.4f}")

    return p, accuracy


# %% [markdown]
# ## 5. Model Training

# %% [code]
if X_train_t is not None and y_train_t is not None and X_val_t is not None and y_val_t is not None:
    # Data (X_train_t, y_train_t, X_val_t, y_val_t) is already prepared and transposed from the preprocessing cell.
    # X_train_t and y_train_t are now the SMOTE'd versions.

    print(f"Confirming shapes before nn_model call:")
    print(f"Shape of X_train_t: {X_train_t.shape}")
    print(f"Shape of y_train_t: {y_train_t.shape}")
    print(f"Shape of X_val_t: {X_val_t.shape}")
    print(f"Shape of y_val_t: {y_val_t.shape}")

    # Define network architecture
    input_dim = X_train_t.shape[0]
    # Example: Input -> Hidden Layer 1 (e.g., 20 units) -> Hidden Layer 2 (e.g., 7 units) -> Output Layer (1 unit for binary classification)
    layers_dims = [input_dim, 20, 7, 1]
    activations = ['relu', 'relu', 'sigmoid']  # ReLU for hidden layers, Sigmoid for output layer

    # Train the model
    # You can adjust learning_rate and num_iterations
    parameters, costs_train, costs_val = nn_model(
        X_train_t, y_train_t,
        X_val_t, y_val_t,
        layers_dims,
        activations,
        learning_rate=0.0075,
        num_iterations=7500,  # Increased num_iterations
        print_cost=True,
        print_cost_every=100  # Print cost every 100 iterations
    )
else:
    print("Skipping model training as data was not loaded/processed or validation data is missing.")
    parameters = None  # Ensure parameters is defined for the next cell
    costs_train, costs_val = [], []  # Ensure these are defined for plotting cell

# %% [markdown]
# ## 6. Evaluation

# %% [code]
if parameters is not None and X_train_processed is not None:  # X_train_processed check might be redundant, X_train_t is better
    # Plot training and validation cost
    if costs_train and costs_val:  # Check if lists are not empty
        plt.figure(figsize=(10, 6))
        plt.plot(np.squeeze(costs_train), label='Training Cost')
        plt.plot(np.squeeze(costs_val), label='Validation Cost')
        plt.ylabel('Cost')
        plt.xlabel('Iterations (x100)')  # Since we print_cost_every 100
        plt.title(f"Learning Curve (LR={0.0075})")  # Make sure LR matches training
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Cost data not available for plotting.")

    # Evaluation data (X_train_t, y_train_t, X_val_t, y_val_t) should already be in the correct
    # format from the preprocessing/SMOTE cell. X_train_t, y_train_t are SMOTE'd.

    print("\n--- Evaluation on Training Data (after SMOTE) ---")
    # Note: y_train_t here is the SMOTE'd target for training data
    Y_prediction_train_labels, _ = predict(X_train_t, y_train_t, parameters, activations)
    print("Classification Report:")
    print(classification_report(y_train_t.flatten(), Y_prediction_train_labels.flatten()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train_t.flatten(), Y_prediction_train_labels.flatten()))

    print("\n--- Evaluation on Validation Data (Original) ---")
    # Note: y_val_t here is the original validation target
    Y_prediction_val_labels, _ = predict(X_val_t, y_val_t, parameters, activations)
    print("Classification Report:")
    print(classification_report(y_val_t.flatten(), Y_prediction_val_labels.flatten()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val_t.flatten(), Y_prediction_val_labels.flatten()))

else:
    print("Skipping model evaluation as the model was not trained or data is unavailable.")
