# Question 1
Larger batch sizes result in better computational efficiency and when experimenting, the training time was lower for the larger batch sizes. This is because modern hardware, especially GPUs, are optimized for parallel processing, and larger batches can exploit this parallelism.

Smaller batch sizes can lead to better generalization. Training with smaller batches introduces more noise into the weight updates, which can help the model generalize better to unseen data. Larger batches resulted in a model that memorizes the training data more, leading to overfitting on the training set and potentially poor generalization to new data.

The figure below shows the results of varying the batch size for the jsc-tiny model over 10 epochs. Batch sizes of 64, 128, 256, and 512.
After 10 epochs, the model with a batch size of 64 has a test accuracy of 49.50%, batch size of 128 has an accuracy of 51.87%, batch size of 256 has an accuracy of 50.48% while a batch size of 512 has an accuracy of 44.64%.

# Question 2
If the maximum number of epochs is too low, the model might not have sufficient time to learn the underlying patterns in the data, leading to underfitting. The model may not perform well on either the training set or unseen data.

However, if the maximum number of epochs is too high, the model may start memorizing the training data (overfitting), capturing noise in the dataset rather than learning the underlying patterns. This can result in poor generalization to new data.

Additonally, increasing the maximum number of epochs increases the overall training time. This is because the model goes through more iterations of updating its weights with each additional epoch.

To show the impact of this, the jsc-tiny model was used with a batch size of 128 over 5, 10, and 20 epochs. 
After 5 epochs, the accuracy is 50.26%, it is 51.87% after 10 epochs and x% after 20 epochs. 

# Question 3
## Large and Small Learning Rate
### large Learning Rate
- Convergence Speed: A large learning rate can lead to faster convergence during training. The model's weights are updated by larger amounts in each iteration, potentially reaching a good solution more quickly.
- Overshooting: However, a too-large learning rate might cause the model to overshoot the optimal weights, leading to oscillations or divergence. The training process can become unstable, and the model might fail to converge to a good solution.
- Risk of Missing Optima: In some cases, a large learning rate might cause the model to skip over optimal regions in the parameter space, preventing it from finding the best solution.

### Small Learning Rate
- Stability: A small learning rate promotes a more stable training process. Smaller updates to the weights help the model fine-tune its parameters gradually, which can be beneficial for convergence.
- Convergence Speed: However, using a very small learning rate might slow down the convergence process. The model may require more epochs to reach an optimal or near-optimal solution.
- Less Sensitivity to Noise: Smaller learning rates make the training process less sensitive to noisy fluctuations in the training data. This can be advantageous for generalization to unseen data.

## Relationship Between Learning Rate and Batch size
Smaller batch sizes often require a smaller learning rate because the model's parameters are updated more frequently, and large learning rate steps may lead to instability.
Larger batch sizes may tolerate higher learning rates because the updates are less frequent, and more training examples are considered in each iteration.

# Question 4

# Question 5


