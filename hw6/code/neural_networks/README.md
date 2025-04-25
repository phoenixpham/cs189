# Phoenix Pham (3037848920)

# How to reproduce the Homework 6 results:
1. Ensure that your Python environment is configured properly with required dependencies.
2. For **5.2**, run the ```train_ffnn.py``` script to generate the training/validation loss and accuracy plots of the neural network model. Ensure that you change the hyperparameters accordingly in ```train_ffnn.py``` to get the specific results of the neural network with the given hyperparameters. The learning rate `lr` is located in the `optimizer_args` attribute dictionary, and the hidden layer size `n_out` is located in the `fc1` attribute dictionary.
3. For **6.1**, run all cells of the ```einsum_examples.ipynb``` file on Jupyter notebook or Google Colab.
4. For **7.1**, run all cells of the ```CS189_HW6_NN_MNIST.ipynb``` file on Google Colab. For faster training, switch your device on Google Colab to GPU. To use a GPU, go to `Runtime` -> `Change runtime type` -> select `GPU`.
5. For **7.2**, run all cells of the ```CS189_HW6_CIFAR10_Transfer_Learning.ipynb``` file on Google Colab. For faster training, switch your device on Google Colab to GPU. To use a GPU, go to `Runtime` -> `Change runtime type` -> select `GPU`. For **7.2.5**, to get the performance on the CNN model with all unfrozen weights, change the parameter `freeze` to `False` when initalizing the model for training. Then rerun the cells that include intializing the model, and the training script. You can also run the following cell to get the loss/accuracy plots.

# How to reproduce the Kaggle results:
1. Run all cells of the ```CS189_HW6_CIFAR10_Transfer_Learning.ipynb``` file on Google Colab to generate the results for the CIFAR-10 dataset. For faster training, switch your device on Google Colab to GPU. To use a GPU, go to `Runtime` -> `Change runtime type` -> select `GPU`. For the `TransferCIFAR10` model, ensure the variable `freeze` is set to `True` (set `True` automatically if you don't pass it as a parameter) in order to get the correct test predictions for Kaggle that uses the model that freezes all the weights in the convolutional layers from the pretrained model.
