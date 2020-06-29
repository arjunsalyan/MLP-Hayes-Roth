This assignment was done as a part of the Course (CSE 464 - Machine Learning), under the mentorship of Dr. Pratik Chattopadhyay.

### Objective
Designing a 3-layer Multilayer Perceptron (Input, Hidden and Output).
Tuning a classifier on a given data set.

### Libraries and Dataset Used
- Dataset: Hayes-Roth (https://archive.ics.uci.edu/ml/datasets/Hayes-Roth)
- Language: Python3
- Third-party libraries: scikit-learn, pandas, numpy

### Procedure/ Steps performed
- The dataset was first normalised before feeding into the model.
- A set of 10 hyperparameters was chosen randomly. Each set included the three layers (input, hidden and output) and the learning rate.
- For each set of hyperparameters, a MLP model was trained with random initialization 20 times. Each time, cross validation accuracy was computed using a k-fold algorithm. The model was trained until convergence.
- For the obtained results, boxplot was prepared.
- The best set of hyperparameters was chosen and a model was designed for these parameters.
- The accuracy of this model was determined using the test dataset.
- A cumulative match characteristics curve was prepared showing the accuracy of the model with increment in ranks.

### Results
**Performance of the model:**

Confusion Matrix:
[[10  3  1]
 [ 3  9  0]
 [ 1  1  6]]

Accuracy Score:  0.735

### Discussion
A node, also called a neuron or Perceptron, is a computational unit that has one or more weighted input connections, a transfer function that combines the inputs in some way, and an output connection.
A single-layer artificial neural network, also called a single-layer, has a single layer of nodes, as its name suggests. Each node in the single layer connects directly to an input variable and contributes to an output variable. A single-layer network can be extended to a multiple-layer network, referred to as a Multilayer Perceptron. A Multilayer Perceptron, or MLP for short, is an artificial neural network with more than a single layer.
Artificial neural networks have two main hyperparameters that control the architecture or topology of the network: the number of layers and the number of nodes in each hidden layer. We must specify values for these parameters when configuring the network. But how do we obtain the optimal set of hyperparameters?

In general, we cannot analytically calculate the number of layers or the number of nodes to use per layer in an artificial neural network to address a specific real-world predictive modeling problem. In our case, we used experimentation to determine what works best for our data set.
