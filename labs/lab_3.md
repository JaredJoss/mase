# Quesion 1
The following quality metrics can be used for the search process:
- Latency: Latency is the time taken by the model to make predictions on a single input. It measures the inference time and is crucial for applications with real-time constraints.
- Model Size: Model size refers to the number of parameters in the neural network. It provides insights into the complexity and memory requirements of the model.
- Number of FLOPs (Floating-Point Operations): FLOPs represent the number of floating-point operations performed by the model during inference. It indicates the computational complexity of the model.
- Precision, Recall, F1 Score: Precision, recall, and F1 score are metrics commonly used in binary or multiclass classification to evaluate the performance of a model on specific classes. They provide insights into the model's ability to correctly identify positive instances (precision), capture all positive instances (recall), and balance between precision and recall (F1 score).
- Resource Utilization (CPU, GPU, Memory): Measure the utilization of computational resources during model inference. This includes CPU usage, GPU usage, and memory usage.
- Energy Consumption: Estimate the energy consumption of the model during inference. This is important for resource-constrained environments and mobile applications.

# Question 2
Accuracy and loss serve as the same quality metric because the task at hand is a classification task, where the primary goal is to correctly classify input samples. Accuracy is a common metric used to measure the model's ability to correctly classify samples, while loss is often used as the optimization objective in training. In classification tasks, optimizing for accuracy and minimizing cross-entropy loss generally align with each other.