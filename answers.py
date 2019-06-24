r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.1, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 answers


def part4_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq, temperature = 'ACT I.', 0.7
    # ========================
    return start_seq, temperature


part4_q1 = r"""
**Your answer:**

We split the corpus to sequences because:
- the entire corpus can't fit in GPU memory all at once

"""

part4_q2 = r"""
**Your answer:**


The text can show memory longer than the sequence length because:
- the hidden state isn't reset after `sequence_length` characters, and can remember further back.
- when training, we didn't reset the hidden state between batches either, just between epochs.

"""

part4_q3 = r"""
**Your answer:**


We do not shuffle the order of batches when training because:
- we assume a relation between the next character to the characters before it, and model it as a hidden state. If we would've shuffled the order, the "characters before the next character" would be random and the hidden state won't reflect text of a real work of art. As result, the network won't be able to learn correctly the parameters that control how the hidden state affects the output. (specifically $W_{hz}$, $W_{hr}$, $W_{hg}$, $W_{hy}$ and the biases)


"""

part4_q4 = r"""
**Your answer:**


1. During training we use a high temperature because we want the probability distribution of "what is the next character" to have a high variance. This allows the network to train against a wider range of predictions, promotes better learning and prevents overfitting.
We lower the temperature for sampling because it means a lower variance, and thus a better chance that the next generated character is actually related to the previous characters (represented as hidden state), as opposed to the next character being random and unrelated.

2. When the temperature is very high, the generated text contains many spelling mistakes and made-up words.
This is because the probability distribution is more uniform and has a higher variance.
Meaning, the next character generated has a higher chance to be unrelated to the previous characters.
Additionaly, the structure of the text looks more like a play because it has many line breaks and capital letters, and also more panctuation.
The has more of those because they are rarer than other characters (e.g. lowercase letters), and thus have a higher chance to be generated when the variance is high.

3. When the temperature is very low, the generated text contains almost zero spelling mistakes or made-up words, but the structure doesn't look like a play. The text also has a tendency to repeat an expression of 2-3 words several times in succession (longer sequences for lower temperatures) before breaking the loop and moving on to other words.
This is because the probability distribution is less uniform and has low variance, and thus is much more deterministic than before.
Basically, this is the opposite of the high-temperature case with parallel reasoning.
We do note that the lower variance supposedly could have caused more spelling mistakes, but this doesn't happen thanks to the memory contained in the hidden state being long enough (more than 3-4 characters back).
The repeating expressions can happen when the hidden state causes a "cycle" and is due to the deterministic nature of the distribution. For example, if the last characters were "the well " and we assume the network and hidden state are such that the most likely next character is "t", and afterwards "h", "e", " ", "w", etc. in a cycle, because of the low variance the most likely next character has a very high likelihood (delta-like) and the cycle will indeed be realized.
This results in the generated text containing a string of "the well the well the well" repeatedly, until the cycle breaks due to a lower-likelihood next character being generated (by chance).


"""
# ==============


# ==============
# Part 5 answers

PART5_CUSTOM_DATA_URL = None


def part5_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 6 answers

PART6_CUSTOM_DATA_URL = None


def part6_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=128, z_dim=100,
        data_label=1, label_noise=0.2,
        discriminator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=0.001,
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.001,
        ),
    )
    # ========================
    return hypers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""