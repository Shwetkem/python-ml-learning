# Neural Network From Scratch

A simple 2-2-1 neural network (2 inputs, 2 hidden neurons, 1 output neuron) built using only NumPy - no ML frameworks - to understand feedforward and backpropagation (the chain rule) from first principles.

**Task:** predict gender (Male/Female) from weight and height, using a toy 4-sample dataset.

This implementation follows [Victor Zhou's neural network tutorial](https://victorzhou.com/blog/intro-to-neural-networks/), done to understand backprop and gradient descent from the ground up rather than as original research.

## What's covered
- Feedforward pass through a hidden layer + output layer
- Manual backpropagation using the chain rule
- Sigmoid activation and its derivative
- Why input features are shifted/centered before training (avoiding sigmoid saturation)
- Stochastic gradient descent, one training example at a time