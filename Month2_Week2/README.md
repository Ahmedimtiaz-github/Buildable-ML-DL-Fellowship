# Month 2 - Week 2 Assignment

## Objectives
- Understand the architecture and components of Convolutional Neural Networks (CNNs).
- Learn about convolutional and pooling layers and their role in feature extraction.
- Apply the knowledge to practical computer vision tasks using real-world datasets.
- Implement and utilize callbacks to improve model training.

---

## Task 1: Understanding CNNs
- **Advantages of Convolutional Layers over Fully Connected Layers**
  - Exploit spatial locality in images.
  - Require fewer parameters → less overfitting.
  - Translation-invariant feature extraction.
  - Enable hierarchical feature learning.

- **How Pooling Reduces Complexity**
  - Downsamples feature maps.
  - Reduces parameters, computation, and memory usage.
  - Adds robustness to small shifts in input.

- **Pooling Types**
  - **Max Pooling:** Keeps strongest activations, good for edges/features, but may lose average context.
  - **Average Pooling:** Smooths features, captures overall statistics, but may dilute strong signals.

> ✅ Implemented in: `Task1_CNN_Concepts.txt`

---

## Task 2: Implementing ANN vs CNN

### Dataset 1: MNIST
- **Implemented Models**
  - Simple **ANN**
  - Simple **CNN**
- **Callbacks Used**
  - `EarlyStopping`
  - `ModelCheckpoint`
- **Comparison**
  - ANN achieves reasonable accuracy but struggles with complex image patterns.
  - CNN converges faster and achieves much higher accuracy.
- **Deliverables**
  - Notebook: `MNIST_ANN_vs_CNN.ipynb`
  - Includes training logs, plots, accuracy/loss curves.

---

### Dataset 2: Cats vs Dogs
- **Implemented Models**
  - Simple **ANN**
  - Simple **CNN**
- **Dataset Source**
  - Loaded using **TensorFlow Datasets (`tfds.cats_vs_dogs`)**
- **Callbacks Used**
  - `EarlyStopping`
  - `ModelCheckpoint`
- **Comparison**
  - ANN performance is limited due to high image complexity.
  - CNN captures spatial features better, achieving higher validation accuracy.
- **Deliverables**
  - Notebook: `CatsVsDogs_ANN_vs_CNN.ipynb`
  - Includes training logs, plots, accuracy/loss curves.

---


---

## Deliverables Summary
- **Notebooks**
  - `Task1_CNN_Concepts.ipynb`
  - `MNIST_ANN_vs_CNN.ipynb`
  - `CatsVsDogs_ANN_vs_CNN.ipynb`
- **Documentation**
  - This `README.md` file.
- **Outputs**
  - Accuracy & loss curves for all models.
  - Observations on ANN vs CNN performance.

## Final Reflection

During this assignment I implemented and compared ANN and CNN models on MNIST and Cats-vs-Dogs. Key takeaways:
- **CNNs** are better for images because they use local receptive fields and weight sharing to learn hierarchical spatial features.
- **Pooling** reduces spatial size and computations while adding small translational invariance.
- **Callbacks** (EarlyStopping, ModelCheckpoint) help prevent overfitting and keep the best model checkpoints.
- **Transfer learning** (MobileNetV2) is an effective strategy to get high accuracy quickly when dataset size, compute, or time are limited.


