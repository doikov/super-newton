# super-newton

Experiments with Super-Universal Regularized Newton Method (see the paper https://arxiv.org/abs/2208.05888
by N. Doikov, K. Mishchenko, and Yu. Nesterov).

1. **Polytope Feasibility** (`experiment_polytope_feasibility.ipynb`):

$$\min\limits_{x \in \mathbb{R}^n} f(x) 
\quad = \quad \sum\limits_{i = 1}^m \max(0,  \langle a_i, x \rangle - b_i\bigr)^p,
\qquad p \geq 2.$$

2. **Soft Max** (`experiment_soft_maximum.ipynb`):

$$
\min\limits_{x \in \mathbb{R}^n} f(x)
\quad = \quad \mu \ln \biggl(  \sum\limits_{i = 1}^m \exp\Bigl( \frac{\langle a_i, x \rangle - b_i}{\mu} \Bigr)\biggr)
\quad \approx \quad \max\limits_{1 \leq i \leq m} \bigl[ \langle a_i, x \rangle - b_i \bigr], 
\qquad \mu > 0.
$$

3. **Worst Instances** (`experiment_worst_instances.ipynb`)

$$
\min\limits_{x \in \mathbb{R}^n} f(x)
\quad = \quad \frac{1}{q} 
\sum\limits_{i = 1}^{n - 1} |x^{(i)} - x^{(i + 1)}|^q + \frac{1}{q} |x^{ (n) }|^q,
\qquad q \geq 2.
$$
