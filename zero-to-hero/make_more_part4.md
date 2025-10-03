# Chapter 4: Becoming a Backprop Ninja 🥋

So far, we've built and trained a multi-layer perceptron, relying on
PyTorch's automatic differentiation engine with a simple call to
`loss.backward()`. While convenient, this function hides a great deal of
complexity. In this chapter, we will peel back that layer of abstraction
and implement the **backward pass manually, from scratch, on the level
of tensors**.

This is a valuable exercise because backpropagation is what's known as a
**"leaky abstraction"**. Simply stacking layers and calling
`.backward()` doesn't guarantee your network will work, or work
optimally. To effectively debug issues like vanishing/exploding
gradients, dead neurons, or saturated activation functions, you need to
understand the internals. A deep understanding can also help you spot
subtle but significant bugs.

A decade ago, writing the backward pass by hand was standard practice.
By recreating this process, we will gain a fully explicit understanding
of our model, emerge with stronger debugging skills, and eliminate any
nervousness about what's hidden away from us.

------------------------------------------------------------------------

## The Setup: An Expanded Forward Pass

We'll use the same two-layer MLP with a batch normalization layer as
before, but with a few key changes to facilitate our manual
backpropagation exercise:

-   **Initialization:** We'll initialize **biases with small random
    numbers** instead of zero. This helps unmask potential bugs in our
    gradient calculations, as zero-initialized variables can sometimes
    simplify gradient expressions in a way that hides errors.
-   **Expanded Forward Pass:** The forward pass is broken down into many
    more intermediate steps. Each intermediate tensor we create will
    have a corresponding gradient tensor in the backward pass, denoted
    with a `d` prefix (e.g., `log\_probs` will have
    $d\text{log\\_probs}$).

### Forward Pass Snippet

\_\_CODEBLOCK0\_\_

Our goal is to manually calculate the gradients for $W1, b1, W2, b2$,
and the batch norm parameters, and all intermediate tensors, *without*
using `loss.backward()`.

------------------------------------------------------------------------

## Exercise 1: The Full Backward Pass, Step-by-Step

We'll backpropagate through the entire network, starting from the loss
and working our way back to the embeddings.

### 1. Loss $\rightarrow$ $\text{log\\_probs}$

The loss is $\text{loss} = -1/n * (\dots)$.

-   **Derivative:** The derivative of the loss with respect to any
    single one of the **correct** log probabilities is **$-1/n$**. The
    gradient for all other elements is **zero**.
-   **Implementation:** $d\text{log\\_probs}$ is initialized to zeros
    with $-1/n$ placed at the indices of the correct characters.

### 2. $\text{log\\_probs} \rightarrow \text{probs}$

This is an element-wise $\log(x)$ operation, where
$d(\log(x))/dx = 1/x$.

-   **Chain Rule:** $d\text{probs}$ is the local derivative
    ($1 / \text{probs}$) multiplied by the incoming gradient
    ($d\text{log\\_probs}$).
-   **Intuition:** This creates a **boosting effect**. If the network
    assigned a very low probability to a correct character,
    $1/\text{probs}$ will be large, amplifying the gradient for that
    error.

### 3. $\text{probs} \rightarrow \text{counts}$ and $\text{count\\_sum\\_inv}$

The operation $\text{probs} = \text{counts} * \text{count\\_sum\\_inv}$
involves **broadcasting** of $\text{count\\_sum\\_inv}$ ($32 \times 1$)
across dimension 1.

-   **Backward Pass Duality:** A **broadcasting** operation in the
    forward pass corresponds to a **summation** in the backward pass
    along the same dimension.
-   \*\*dcount\\\_sum\\\_inv:\*\* Its gradient must be the **sum**
    across dimension 1 of the gradient from the multiplication.

### 4. The exp and Numerical Stability Layers

The steps include exponentiation and the numerical stability offset
($\text{norm\\_logits} = \text{logits} - \text{logit\\_maxes}$).

-   $d\text{norm\\_logits}$: Since
    $\text{counts} = \exp(\text{norm\\_logits})$,
    $d\text{norm\\_logits} = \text{counts} * d\text{counts}$.
-   $d\text{logit\\_maxes}$: Backpropagating through the subtraction
    requires a **sum** for $d\text{logit\\_maxes}$ due to broadcasting,
    though its value should be near zero.

### 5. The Linear Layer (Matrix Multiplication and Bias)

Backpropagating through a linear layer ($\text{logits} = h @ W2 + b2$).
$N$ is the batch size, $D\_{\text{in}}$ is the input dimension, and
$D\_{\text{out}}$ is the output dimension.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Gradient Target         Formula                                                   Shape Calculation
  ----------------------- --------------------------------------------------------- ------------------------------------------------------------------------------------------------------------
  **$d\text{h}$**         $d\text{h} = d\text{logits} @ W2^T$                       $(N \times D\_{\text{out}}) @ (D\_{\text{out}} \times D\_{\text{in}}) \rightarrow N \times D\_{\text{in}}$

  **$d\text{W2}$**        $d\text{W2} = h^T @ d\text{logits}$                       $(D\_{\text{in}} \times N) @ (N \times D\_{\text{out}}) \rightarrow D\_{\text{in}} \times D\_{\text{out}}$

  **$d\text{b2}$**        $d\text{b2} = d\text{logits}.\text{sum}(\text{axis}=0)$   $1 \times D\_{\text{out}}$
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

> **The Dimensionality Trick** 🧠 You can derive these formulas by
> simply matching dimensions. For **$d\text{W2}$**, you must multiply
> $h$ and $d\text{logits}$ to get the shape of $W2$. The *only* way to
> arrange the tensors for valid matrix multiplication is
> $h^T @ d\text{logits}$. This trick consistently yields the correct
> result.

### 6. Tanh Activation

The derivative for $\tanh(x)$ is $1 - \tanh(x)^2$. Since $h$ is the
output of the $\tanh$, the local derivative is $1 - h^2$. We apply the
chain rule: $d\text{h\\_preact} = (1 - h^2) * d\text{h}$.

### 7. Batch Normalization Layer

This complex layer is broken down into atomic operations. We
backpropagate through each, paying close attention to broadcasting,
which necessitates summation.

-   **Bessel's Correction:** We use the **unbiased estimator** ($n-1$)
    in our variance calculation for consistency.

### 8. Embedding Layer

The operation is an indexing lookup: $\text{emb} = C[Xb]$.

-   **Backward Pass:** The gradient $dC$ is initialized to zeros. For
    each gradient in $d\text{emb}$, we **add** it to the corresponding
    row in $dC$ ($dC[Xb[i]] += d\text{emb}[i]$). The addition (`+=`) is
    crucial because a single embedding row may be used multiple times in
    a batch.

------------------------------------------------------------------------

## Exercise 2: An Efficient Shortcut for the Loss

The step-by-step backpropagation through the cross-entropy loss is
inefficient. Analytical derivation shows the intermediate terms cancel,
providing a single, simple, and efficient formula for
$d(\text{loss})/d(\text{logits})$.

### The Analytical Gradient for dlogits

1.  Calculate the probabilities: $p = \text{softmax}(\text{logits})$.
2.  Subtract **1** from the probabilities at the indices of the correct
    labels.
3.  Divide the entire result by the batch size, $n$.

\_\_CODEBLOCK1\_\_

**Intuition:** The gradient acts like a system of forces. It "**pulls
up**" on the logit for the correct class and "**push down**" on the
logits for all incorrect classes, with strength proportional to the
model's assigned probability.

------------------------------------------------------------------------

## Exercise 3: Analytical Gradient for Batch Normalization

We can derive a single, highly efficient (though algebraically complex)
expression for $dX$ from $dY$ for the entire Batch Normalization layer.

This involves tracking contributions from the three paths: direct
influence, and indirect influence through the mean ($\mu$) and the
variance ($\sigma^2$), and summing their effects.

------------------------------------------------------------------------

## Exercise 4: Putting It All Together

We assemble our efficient, manually-derived gradient functions into a
complete training loop. We replace the call to `loss.backward()` with
our own gradient computation logic.

The entire forward and backward pass can be wrapped in a
`with torch.no\_grad():` block for added efficiency, as we are handling
the gradient computation ourselves.

The model trains to the same performance, but now we have **full
visibility and understanding** of what is happening under the hood. You
now have the deep intuition of how gradients flow from the loss back
through every parameter of the network. **You are a backprop ninja.**
