# Neural Networks and Deep Learning

A beginner-friendly introduction to neural networks. No prior machine learning experience is required. We start with the math behind a single neuron and build intuition with NumPy and Jupyter before moving on to full networks in later modules.

---

## What is in this repo

| File | What it is |
|------|------------|
| `Module_01_Mathematical_Foundations_of_Neural_Networks.ipynb` | Module 1 lab — read, run, and experiment |
| `requirements.txt` | Python packages you need |

---

## Getting started

1. **Python 3.10+** installed on your machine.
2. Open a terminal in this folder and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter Lab:

   ```bash
   jupyter lab
   ```

4. Open the Module 1 notebook and run cells from top to bottom (`Shift + Enter`). Change numbers in the code and re-run — that is how the ideas stick.

---

## Module 1 — what you will learn (in plain English)

Module 1 answers one question: *“What is actually happening inside one neuron?”* Everything below is covered in the notebook with examples, plots, and code you can run.

### Part 1 — Neurons and perceptrons

**Biological neuron (intuition only)**  
Your brain has cells that listen to signals from other cells. When enough signal builds up, the cell “fires” and passes something on. We are not simulating biology in code — we borrow one idea: combine many inputs, then decide how strongly to pass a signal onward.

**Artificial neuron**  
The building block of every neural network. It:

1. Takes numbers as **inputs** (e.g. exam scores, pixel brightness).
2. Multiplies each input by a **weight** — how much that input matters (learned during training).
3. Adds a **bias** — a knob that shifts the overall score up or down.
4. Runs the result through an **activation function** to produce the **output**.

In words: *listen to each input with different importance, add a baseline shift, then shape what you send out.*

**Perceptron**  
An old name (1958) for a single artificial neuron. People still say “perceptron” when they mean one neuron, especially with a simple on/off style activation. Modern networks are just many neurons stacked in **layers**.

| Term | Plain meaning |
|------|----------------|
| Input | What you feed in |
| Weight | How strongly each input counts |
| Bias | Extra offset before the decision |
| Activation | The function that shapes the output |
| Output | What this neuron sends to the next step |

---

### Part 2 — Weighted sum

Before activation, the neuron computes a **score** by multiplying each input by its weight and adding everything up.

Example: if humidity matters a lot and wind matters a little, humidity’s weight is large and wind’s might be small or negative (pushing the score down).

**Dot product** is the fast way to do the same thing in code: `np.dot(w, x)` or `w @ x` in NumPy. That score is often called **pre-activation** or **logit** (you will hear “logit” again with softmax in classification).

---

### Part 3 — Bias

Without bias, if all inputs are zero, the weighted sum is always zero — no matter what the weights are. That is too rigid.

**Bias** is one extra number added after the weighted sum:  
`score = (weights · inputs) + bias`

- **Weights** control how sensitive you are to each input (slope).
- **Bias** shifts the whole decision up or down (offset).

Example: predicting pass/fail from study hours. The weight on hours says “more hours → higher score.” The bias says “even at zero hours, the baseline isn’t forced to be exactly zero” (or the opposite, depending on what the model learned).

Some courses teach bias as “a fake input that is always 1 with its own weight.” Same idea, different notation. The notebook uses explicit `b` because it is clearer when you are starting out.

---

### Part 4 — Activation functions

If you only used `output = score` with no extra function, stacking many neurons would still behave like **one big linear formula**. Real-world patterns (images, language, etc.) are not that simple — you need **nonlinearity**.

An **activation function** takes the pre-activation score `z` and returns `a = f(z)`. Module 1 covers five common ones:

#### Sigmoid
Squashes any number into a range between 0 and 1 — handy when you want something that feels like a probability.

- **Good for:** binary yes/no outputs (e.g. “will it rain?”).
- **Watch out for:** very large positive or negative scores saturate (learning slows — “vanishing gradients”); outputs are not centered around zero.

#### Tanh (hyperbolic tangent)
Similar to sigmoid but outputs between **-1 and 1**, centered around zero.

- **Good for:** hidden layers in older designs; some recurrent networks.
- **Watch out for:** still saturates at the extremes.

#### ReLU (Rectified Linear Unit)
If the score is positive, pass it through unchanged. If zero or negative, output zero.

- **Good for:** default choice in most modern hidden layers — fast, simple, helps deep networks train.
- **Watch out for:** **dying ReLU** — if a neuron always gets negative scores, it can output zero forever and stop learning.

#### Leaky ReLU
Like ReLU, but when the score is negative you allow a **small** negative slope instead of hard zero (e.g. 1% of the score). That gives the neuron a way to recover if it got stuck at zero.

#### Softmax
Used when you have **several classes** and want **probabilities that add up to 100%** (e.g. digit 0–9, or Apple / Cherry / Banana). It takes a vector of scores (one per class) and turns them into a probability distribution. The class with the highest score gets the highest probability.

The notebook also plots all of these side by side so you can *see* the difference, not just memorize formulas.

---

### Part 5 — Forward propagation

**Forward propagation** means: data flows from **input → output** through the network. No learning yet — you are only *using* the current weights.

For **one neuron**, the pipeline is always:

1. **Linear step:** `z = weights · inputs + bias`
2. **Activation step:** `a = activation(z)`
3. Send `a` to the next layer, or use it as the final prediction.

In a **deep** network, layer 1’s outputs become layer 2’s inputs, and so on. **Training** (later modules) uses **backpropagation** to adjust weights and bias; forward propagation is just the forward pass with whatever weights you have now.

---

### Part 6 — Hands-on with NumPy

You implement the same math libraries like PyTorch and TensorFlow use under the hood — just written clearly:

- A **`forward_one_neuron`** function that prints each step (score, then activation).
- A **`SingleNeuron`** class that stores weights and bias and runs `forward(x)` with your choice of activation.

**Mini example — sigmoid “will it rain?”**  
Two inputs: humidity and cloud cover. Hand-set weights (in real projects, training finds them). Dry and clear → low rain probability; humid and cloudy → high probability.

**Softmax output layer**  
For multiple classes, you often have one weight row per class, compute a score per class, then apply softmax once. That is forward propagation for the **output layer** of a classifier.

---

### Part 7 — Practice exercises

Try these yourself before peeking at the solutions cell:

1. Weighted sum with chosen `x` and `w` — which input actually mattered?
2. Find a bias that makes the pre-activation score exactly zero.
3. ReLU by hand for a few values, then check with code.
4. Softmax on equal logits — should give equal probabilities.
5. Build a `SingleNeuron` with four inputs and `tanh`, then call `forward`.

---

## Quick reference (after Module 1)

| Idea | What to remember |
|------|------------------|
| Weighted sum | `z = w @ x` |
| With bias | `z = w @ x + b` |
| Forward pass | linear step, then activation |
| Sigmoid | Squeezes to (0, 1) — binary-style outputs |
| Tanh | Squeezes to (-1, 1) — zero-centered |
| ReLU | `max(0, z)` — modern default for hidden layers |
| Leaky ReLU | Small slope when z ≤ 0 — helps dead neurons recover |
| Softmax | Scores → probabilities that sum to 1 |

---

## What comes next

Future modules will cover:

- **Loss functions** — how wrong the prediction is
- **Backpropagation** — how the network learns from mistakes
- **Training loops** — many neurons, many layers, real datasets

Module 1 is the foundation. Once one neuron makes sense, stacking thousands of them is mostly the same idea repeated with better organization.

---

## Tips for learning

- Run every code cell yourself; do not only read the markdown.
- When a plot appears, pause and ask: *“What would happen if z were very negative? Very positive?”*
- If something feels abstract, change one weight or bias in the notebook and re-run — you should see the output move in a sensible direction.

When you get stuck: note the cell that confused you, what you expected, and what you got. Re-run from there — that is the fastest way to debug your understanding.
