The Standard Normal Distribution
=======================

In statistical analysis, we use probability to describe how likely a particular result (e.g. a sample mean) is relative to something else (e.g. the population mean). To do this, we will return to the concept of distributions. Many climate and geophysical variables have distributions with similar characteristics, so in this course, we will focus on this type of distribution, the **normal distribution**.

## Probability Density Functions

In the previous section on distributions, we used histograms to visualize the *discrete* probability distributions of surface air temperature data at the UTSC weather station. However, most variables that we will be working with are not discrete, they are *continuous*, and thus, we need a way to describe continuous probability distributions (see the [following][pdf] for more on discrete versus continuous probability distributions).

Let's define a *probability density function* (PDF), a function that describes a probability distribution for a continuous variable, as $f(x)$.

The probability of an event $x$ occurring between values $a$ and $b$ is then,

$$
Pr(a \leq x \leq b) = \int_{a}^b f(x)dx
$$

where $f(x)$ is the **PDF** and has the following properties:

- $f(x) \geq 0$ for all $x$
- $\int_{-\inf}^{\inf} f(x)dx = 1$

Note the similarities between the relative frequency plotted in our histograms and the PDF, both are always greater than or equal to zero and the area under both curves sums to 1 (Note that for our histograms, an unstated assumption was that each bin had a width of 1).

### Cumulative Density Functions

The **cumulative density function** (CDF) is analogous to the *discrete* cumulative probability distribution. The CDF at value $b$, denoted $F(b)$, is the probability that $x$ assumes a value less than $b$

$$
\begin{align}
Pr(x \leq b) &= F(b)
&= \int_{-\inf}^{b} f(x)dx
$$

So, for all $a$ $\leq$ $b$, we can show that

$$
Pr(a \leq x \leq b) = F(b) - F(a)
$$

## The Normal Distribution

So far, we have described probability density functions in general terms. Now, we will take a closer look at the **normal distribution**.

The PDF of a variable $x$ that is *normally* distributed about its mean value is given by

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}
$$

Let's take a look at what this looks like for a given set of parameters.


The associated cumulative distribution function is

$$
F(b) = \int_{-\inf}^{b} \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}dx
$$

and the probability that $x$ will fall between two values $a$ and $b$ is thus

$$
F(b) - F(a) = \int_{a}^{b} \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}dx
$$





[pdf]: https://www.themathdoctors.org/from-histograms-to-probability-distribution-functions/
