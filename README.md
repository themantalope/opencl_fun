<head>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
</head>

# A repo for a small (py)opencl project

I will be implementing stochastic gradient descent for logistic regression using OpenCL.

-----

### OLS Gradient descent algorithm

- OLS = ordinary least squares
- The objective of this algorithm is to estimate model parameters by minimizing
the sum of squares between the model estimates and training examples
- Cost function:

$$ J(x,y,\beta) = \frac{1}{2} \sum_{i=1}{N} (h(x_{i}, \beta) - y_{i})^{2} $$

Where $h(.)$ is the hypothesis function (a.k.a. model).

- Since this  is a convex function, this is minimized for the parameters $\beta$
when:

$$
\frac{\partial J}{\partial \beta} = 0 \newline
\sum_{i=1}^{N} (h(x_{i}, \beta)) * \frac{\partial h(x_{i}, \beta)}{\partial beta} = 0
$$
