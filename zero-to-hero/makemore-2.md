## Part 1: A Better-Rendered Tutorial: From Concept to Code

This is a structured, step-by-step guide to building the character-level Multi-Layer Perceptron (MLP), following the logic presented in the video.

### **Step 1: The Core Idea - Moving Beyond Bigrams**

The starting point is recognizing the limitations of a simple bigram model, which only uses one previous character as context. Expanding this approach to use more context (e.g., three characters) causes the table of possibilities to grow exponentially (from 27 to nearly 20,000), making it unmanageable. This is known as the **curse of dimensionality**. The MLP solves this by learning **distributed feature representations** for characters, allowing it to generalize.

### **Step 2: Building the Dataset**

The first practical step is to create the dataset. We define a **`block_size`** to specify the context length (e.g., 3 characters to predict the 4th). The code then iterates through the words, creating input-label pairs **`(X, y)`**.

* **Input `X`**: A tensor where each row is a sequence of character indices (e.g., `<.`, `.` for the start of a word).
* **Label `y`**: A tensor containing the index of the character that follows each input sequence.

### **Step 3: Implementing the MLP Architecture**

The model consists of four main parts, based on the Bengio et al. paper:

1.  **Embedding Layer**: We create a lookup table `C`, which is a `27 x D` matrix (where `D` is the embedding dimension, e.g., 2 or 10). For each integer in our input tensor `X`, we look up its corresponding D-dimensional vector. PyTorch's powerful indexing allows us to do this for the entire batch at once with `C[X]`, which is highly efficient.
2.  **Hidden Layer**: The embedding vectors for each context character are concatenated (or more efficiently, reshaped using `.view()`) into a single vector. This vector is then fed into a standard fully connected layer with a `tanh` non-linearity: `h = torch.tanh(emb.view(...) @ W1 + b1)`.
3.  **Output Layer**: The hidden layer's activations are passed through a final linear layer to produce **logits**—a raw score for each of the 27 possible next characters.
4.  **Loss Calculation**: Instead of manually implementing softmax and negative log-likelihood, we use PyTorch's `F.cross_entropy(logits, y)`. This single function is more efficient, numerically stable, and has a simpler backward pass.

### **Step 4: Training with Mini-Batches**

Training on the full dataset of 228,000+ examples at once is extremely slow. The standard practice is to use **mini-batches**. In each iteration, we:

1.  Select a small, random subset of the data (e.g., 32 examples).
2.  Perform the forward pass to calculate the loss on this mini-batch.
3.  Perform the backward pass (`loss.backward()`) to compute gradients.
4.  Update the parameters using the gradients.

This is much faster because an approximate gradient from a small batch is "good enough" to make progress, allowing for far more updates in the same amount of time.

### **Step 5: Proper Evaluation with Data Splits**

To ensure the model is generalizing and not just memorizing the training data, we split the dataset into three parts: **training (80%), validation/dev (10%), and test (10%)**.

* The **training set** is used to update the model's parameters.
* The **validation set** is used to tune hyperparameters (like embedding size or learning rate).
* The **test set** is used only at the very end to get a final, unbiased performance score.

If the training and validation losses are roughly equal, it indicates the model is **underfitting**, meaning it has the capacity to be made larger to improve performance.

---

## Part 2: Rendering a Deeper Understanding of Key Decisions

The video makes several crucial implementation choices for performance and stability. Here is a deeper explanation of *why* those choices are correct.

### **Efficient Input Shaping: `view` is Better Than `cat`**

After the embedding lookup, we have a tensor of shape `(batch_size, block_size, embedding_dim)`, e.g., `(32, 3, 2)`. To feed this into a linear layer, we need to reshape it to `(32, 6)`.

* **The Inefficient Way (`torch.cat`)**: One could concatenate the three 2-dimensional vectors for each example. However, this is inefficient because it creates a brand-new tensor in memory.
* **The Efficient Way (`torch.view`)**: The `.view(32, 6)` operation is extremely fast. This is because PyTorch tensors have an underlying one-dimensional `storage` in memory. The `.view()` operation does not move or create new data; it simply changes the tensor's internal metadata (like shape and strides) that dictates how that 1D block of memory is logically interpreted.

### **Numerically Stable Loss: The Power of `F.cross_entropy`**

You should always prefer `F.cross_entropy` over implementing the loss function manually for three main reasons:

1.  **Forward/Backward Pass Efficiency**: PyTorch uses fused kernels for common operations like cross-entropy, meaning the sequence of exponentiation, summation, and log is performed in a single, highly optimized step without creating large intermediate tensors in memory. The analytical derivative for the backward pass is also much simpler and more efficient than backpropagating through each separate operation.
2.  **Numerical Stability**: Exponentiating logits can cause numerical overflow. If a logit is a large positive number (e.g., 100), `exp(100)` will result in `inf`, leading to `NaN` in the loss. `F.cross_entropy` solves this with a clever trick: it finds the maximum logit value and subtracts it from all logits before exponentiating. This mathematical transformation does not change the final probabilities but ensures the largest input to `exp()` is 0, preventing overflow.

### **The True Power of Embeddings: Generalization**

The core reason the MLP works so much better than the bigram model is its ability to **generalize through the embedding space**. Suppose the model has seen "a cat was running" and "the dog was running" but has never seen "a dog was running".

* By processing many examples, the model learns that the characters (or words) `a` and `the` often appear in similar contexts and can be interchangeable.
* Through backpropagation, it nudges their embedding vectors to be close to each other in the embedding space.
* Therefore, when it encounters the new context "a dog...", the network can leverage its knowledge about the embedding for `the` (which is near `a`) and correctly predict what might come next. This ability to share statistical strength among similar inputs is what allows the model to handle novel combinations it has never seen before. The video shows this happening, as the learned embeddings for vowels (`a, e, i, o, u`) end up clustered together.

---

## Part 3: Rendering a Better Model: Your Roadmap to Improvement

The video achieves a final validation loss of around **2.17** and explicitly provides a roadmap for you to "beat this number". Here are the key knobs you can turn for better performance.

### **1. Systematically Tune Hyperparameters**

* **Find the Optimal Learning Rate**: Don't guess the learning rate. A good technique is to test a range of exponentially spaced learning rates (e.g., from `10^-3` to `10^0`). Plot the resulting loss against the learning rate exponent. The best learning rate is typically in the "valley" of the plot, just before the loss becomes unstable or explodes.
* **Use a Learning Rate Decay**: Start with a good learning rate (e.g., `0.1`) and train for a large number of steps. When the loss plateaus, decay the learning rate (e.g., by a factor of 10 to `0.01`) and continue training to achieve finer convergence.

### **2. Scale Up the Model's Capacity**

The initial model was **underfitting**, meaning it was too small to capture all the complexity in the data. To fix this, you can scale it up:

* **Increase Embedding Dimensionality**: The 2-dimensional embedding was identified as a potential performance bottleneck. Increasing it to 10D or more gives the model a richer space to represent characters.
* **Increase Hidden Layer Size**: A larger hidden layer (e.g., from 100 to 200 or 300 neurons) increases the model's capacity to learn complex patterns between the input embeddings.
* **Increase Context Length (`block_size`)**: Using a larger context (e.g., 8 characters instead of 3) gives the model more information to make its prediction.

### **3. Refine the Optimization Process**

* **Adjust Batch Size**: The video uses a small batch size of 32. A larger batch size can provide a more stable gradient and might lead to better convergence, although each step will be slower. Experimenting with this is another way to improve performance.
