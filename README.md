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














# Lab – 3: Variational Autoencoder (VAE)

**Course:** CSET419: Introduction to Generative AI  
**Experiment:** Implement a Variational Autoencoder (VAE) for Fashion Item Generation

##  Objective
The objective of this experiment is to implement and analyze a **Variational Autoencoder (VAE)**. The goal is to design Encoder-Decoder architectures that learn a probabilistic latent space, allowing for both the efficient reconstruction of input data and the generation of diverse new synthetic samples by sampling from a learned distribution.

##  Experiment Overview
In this experiment, we explore the capabilities of VAEs in learning **continuous, structured representations** of data. Unlike standard autoencoders which simply memorize inputs, VAEs learn a "smooth" latent space. The task involves training a model on fashion items to understand distinct features (e.g., the structural difference between a boot and a sneaker) and utilizing this understanding to generate novel clothing designs from random noise.

##  Domain & Model
* **Domain Selected:** **Fashion-MNIST**
    * A dataset of 28x28 grayscale images representing 10 categories of clothing and accessories.
* **Generative Model:** **Variational Autoencoder (VAE)**
    * **Probabilistic Encoder:** Maps inputs to a distribution (defined by mean $\mu$ and variance $\sigma$).
    * **Decoder:** Reconstructs images from sampled latent vectors.

##  Methodology
1.  **Dataset Preparation:** Loaded and preprocessed the Fashion-MNIST dataset.
2.  **Architecture Design:**
    * Implemented an **Encoder** to compress input images into a lower-dimensional latent distribution.
    * Applied the **Reparameterization Trick** to enable backpropagation through stochastic sampling.
    * Implemented a **Decoder** to reconstruct the original image from the sampled latent vectors.
3.  **Training Process:** Trained the model by minimizing a **dual loss function**:
    * *Reconstruction Loss:* Ensures the output visually resembles the input.
    * *KL Divergence:* Regularizes the learned distribution to approximate a standard normal distribution.
4.  **Analysis & Visualization:**
    * **Reconstruction Analysis:** Plotting original inputs against reconstructed versions to check data compression quality.
    * **Generative Sampling:** generating completely new images by sampling random noise vectors ($z \sim \mathcal{N}(0,1)$).
    * **Latent Space Visualization:** Projecting the test dataset into a 2D latent space to visualize clustering of class labels.

##  Tools & Technologies
* **Language:** Python
* **Platform:** Google Colab
* **Framework:** PyTorch
* **Visualization:** Matplotlib

##  Key Outputs
1.  **Reconstruction Grid:** A visual comparison of original Fashion-MNIST images (Top row) vs. VAE-reconstructed counterparts (Bottom row) to verify feature retention.
2.  **Generative Samples:** A grid of distinct, newly generated fashion items created solely from random noise.
3.  **Latent Space Scatter Plot:** A 2D scatter plot where data points are colored by class label, demonstrating how the model groups semantically similar items (e.g., T-shirts vs. Trousers) in the latent space.

##  Conclusion
This experiment validates the effectiveness of Variational Autoencoders in learning structured data representations. The reconstruction results demonstrate the model's ability to capture essential features, while the generative samples prove it can synthesize novel data points. Furthermore, the latent space visualization confirms that the VAE successfully clusters semantically similar items, proving its utility for unsupervised learning and creative generation tasks.
