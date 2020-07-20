# Summary Statistics: Part I

To get started, we will review several summary statistics that will likely be familiar to you: mean, median, standard deviation, skewness and kurtosis.


## 1. First, a few definitions and a bit of nomenclature...

Throughout the course, we will define $X$ as a random variable (e.g. surface air temperature). The "population" of $X$ represents all possible values of $X$. The true population mean and standard deviation of this variable are denoted as:

$$
\begin{align}                                  
\mu &= \text{population mean}\\                
\sigma &= \text{population standard deviation} 
\end{align}                                     
$$  

A sample of $X$ is a subset of values of $X$. We will define a random sample of $X$ with $N$ elements as [$x_1$, $x_2$, $x_3$, ..., $x_N$]. The sample mean and standard deviation are denoted as:

$$
\begin{align}
\overline{x} &= \text{sample mean}\\
s &= \text{sample standard deviation}
\end{align}
$$



## 2. Moments

Moments are quantitative parameters that we will use to describe a sample of $X$.

### 2.1 The Mean

The first moment is the **mean**. The sample mean is defined as:

$$  
\overline{x} = \frac{1}{N}\sum_{i=1}^{N}x_i 
$$

If we assume that our sample is a measurement taken at a fixed location but at a certain frequency in time (aka, a time series) then the overbar denotes the sample time mean and the subscript i denotes the time step.

```{note} The sample mean is an *unbiased* estimate of the population mean, $\mu$.

In other words, for an infinite number of samples from the same time series, the population mean of all of the sample means ($\mu_{\overline{x}}$) is equal to the population mean ($\mu$). 
```

The **median** is the value in the centre of a population or sample. This is a useful statistic when there are large outliers in the population/sample (e.g. house prices, salaries, rain fall rate, earthquake magnitude)


### 2.2 Variance and Standard Deviation

The second moment of a sample $X$ is the **variance**. As you can probably guess, the variance of a sample of $X$ describes the *variability* of $X$. The sample variance is defined as:

$$  
\overline{x^{\prime 2}} = \frac{1}{N-1}\sum_{i=1}^{N}(x_i - \overline{x})^2
$$

where

$$
x^{\prime}_i = x_i - \overline{x}
$$

Again, assuming that our sample is a time series, then $x^{\prime}_i$ represents the deviations about the sample mean. 

The variance is in squared units for whatever our variable $X$ is. To get an estimate of the variability of $X$ from our sample in the correct units, we use the sample **standard deviation**:

$$
s = \sqrt{\overline{x^{\prime 2}}}
$$

```{note} The sample variance is an *unbiased* estimate of the population variance, $\sigma^2$.

But notice that we are dividing by $N-1$ instead of $N$. Since the calculation of the sample variance depends on the sample mean (which is itself an approximation and introduces some bias), then we need to divide by $N-1$ rather than $N$ to remove this bias. You can find out more [here](https://www.youtube.com/watch?v=D1hgiAla3KI).
```