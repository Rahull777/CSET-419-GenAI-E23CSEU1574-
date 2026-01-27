# Lab – 1

## CSET419: Introduction to Generative AI

### Objective
The objective of this experiment is to understand the fundamentals of **Generative AI** by generating synthetic medical imaging data using a pre-trained generative model. Students will learn how to use Python-based tools to generate, store, and visualize synthetic X-ray samples.

### Experiment 1: Implement a Simple Generative Algorithm for Medical Data Generation

Generative AI models are capable of creating new and realistic data samples based on learned patterns. In this experiment, a **pre-trained generative model** is used to generate synthetic X-ray images, which are then stored in a structured dataset format for further analysis, such as training diagnostic models.

### Domain Selected
Medical Image Generation (Chest/Bone X-Rays)

### Generative Model Used
**Stable Diffusion**
(Pre-trained diffusion-based generative model adapted for medical imaging prompts)

### Methodology / Procedure
1. Selected **Medical X-ray Imaging** as the data domain.
2. Loaded a **pre-trained Stable Diffusion model** using Python.
3. Provided multiple **medical textual input prompts** (minimum of 5) to guide the generation process (e.g., "Frontal chest X-ray of healthy lungs", "X-ray of a fractured radius bone").
4. Generated synthetic X-ray images using the generative model.
5. Saved the generated images in a **folder-based dataset structure**.
6. Displayed sample generated outputs for visual verification of anatomical structure.

### Tools & Technologies
* Python
* Google Colab
* Stable Diffusion
* PyTorch
* Hugging Face Diffusers Library

### Output
* A **synthetic X-ray dataset** generated using a pre-trained generative AI model.
* Images stored in an **organized folder structure**.
* Sample outputs displayed to verify data quality and anatomical realism.

### Conclusion
This experiment demonstrates how **Generative AI models** can be used to efficiently create synthetic medical datasets. The generated data can be further utilized for tasks such as **training diagnostic machine learning models**, **data augmentation for rare conditions**, and experimentation, especially in medical scenarios where real-world patient data is privacy-sensitive or limited.










# Lab – 2

## CSET419: Introduction to Generative AI

### Objective
The objective of this experiment is to implement and train a basic **Generative Adversarial Network (GAN)** to generate new synthetic images. [cite_start]Students will learn to design Generator and Discriminator architectures, manage the adversarial training process, and evaluate the quality of generated samples over multiple training epochs[cite: 19, 41].

### Experiment 2: Train a Basic GAN Model for Image Generation

[cite_start]In this experiment, we address a simulated scenario where a university’s digital archive has partially crashed, resulting in missing image data[cite: 21]. [cite_start]To restore testing capabilities for the AI pipeline, the task is to train a GAN that learns the distribution of a specific dataset and produces realistic-looking synthetic samples to simulate the lost data[cite: 22, 23].

### Domain Selected
[cite_start]Handwritten Digit Generation (MNIST Dataset) [cite: 26]

### Generative Model Used
**Basic Generative Adversarial Network (GAN)**
(Comprising a Generator network to create images from noise and a Discriminator network to distinguish real from fake images) [cite_start][cite: 39, 40]

### Methodology / Procedure
1.  [cite_start]**Dataset Selection:** Selected the **MNIST** dataset (handwritten digits) and normalized images to the range [-1, 1] for stable GAN training[cite: 26, 45].
2.  **Architecture Design:**
    * [cite_start]Designed a **Generator** network to convert random noise vectors into 28x28 pixel images[cite: 39, 46].
    * [cite_start]Designed a **Discriminator** network to classify input images as either "Real" or "Fake"[cite: 40].
3.  [cite_start]**Adversarial Training:** Implemented an alternating training loop where the Generator tries to fool the Discriminator, while the Discriminator tries to correctly identify fake images[cite: 41].
4.  [cite_start]**Monitoring:** Computed loss values (`D_loss`, `G_loss`) and accuracy (`D_acc`) to track convergence[cite: 47, 53].
5.  [cite_start]**Data Generation:** Periodically generated and saved synthetic sample grids during training to visualize progress[cite: 42, 56].
6.  [cite_start]**Evaluation:** Generated a final batch of 100 synthetic images and used a pre-trained classifier to predict and analyze their labels[cite: 43, 58].

### Tools & Technologies
* Python
* Google Colab
* [cite_start]TensorFlow / Keras [cite: 28]
* Matplotlib (for visualization)

### Output
*Training Logs: Detailed epoch-wise logs showing Generator Loss, Discriminator Loss, and Accuracy.
* Generated Samples:A folder structure containing image grids saved at regular intervals (`generated_samples/`).
*  Dataset:A collection of 100 final synthetic images stored in `final_generated_images/`[cite: 58].
*  Analysis:A distribution report of the generated images based on predictions from a pre-trained classifier.

### Conclusion
This experiment successfully demonstrates the capability of **Generative Adversarial Networks (GANs)** to learn complex data distributions without direct supervision. By balancing the competition between the Generator and Discriminator, the model evolved from producing random noise to generating recognizable handwritten digits. This validates the potential of GANs for **synthetic data generation**, which is critical for restoring lost datasets and testing AI pipelines when real data is unavailable.
