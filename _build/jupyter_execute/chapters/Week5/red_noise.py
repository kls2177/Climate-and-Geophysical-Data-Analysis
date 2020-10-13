#!/usr/bin/env python
# coding: utf-8

# # Introduction to Red Noise
# 
# To determine by how much we need to decrease $N$ to account for persistence in a time series, we will use a theoretical model for an autocorrelated time series, called a **red noise** time series.
# 
# If you scroll down, you will see a lot of equations. *Don't panic*. The math is actually not that bad. The equations actually help to demonstrate that red noise time series are simpler than you might first think. Let's start with the definition of a red noise time series.
# 
# 
# ## Definition of Red Noise
# 
# A red noise time series, $x(t)$, is defined mathematically as:
# 
# $$
# x(t) = ax(t - \Delta t) + b\epsilon(t)
# $$
# 
# where, 
# 
# - $x$ is a standardized variable
# - $\Delta t$ is the time interval between adjacent data points (and is assumed to be a constant)
# - $a$ lies between 0 and 1 and measures the *memory* of the previous state, i.e., the larger the value of $a$, the more alike adjacent data points are.
# - ($t - \Delta t$) is the time before time $t$
# - $\epsilon(t)$ is a random variable drawn from the standard normal distribution and represents *white noise* in the system (we will formally define white noise later)
# 
# ### Coefficients of a Red Noise Time Series
# 
# To determine $a$, we multiply the both sides of the equation above by $x(t - \Delta t)$ and take the time average,
# 
# $$
# \overline{x(t)x(t - \Delta t)} = a\overline{x(t - \Delta t)x(t - \Delta t)} + b\overline{\epsilon(t)x(t - \Delta t)}
# $$
# 
# Since $x$ is standardized (variance of 1), the first term of the right-hand-side (rhs) of the equation is $a$ x 1. In addition, since $\epsilon(t)$ is random in time, assuming your time series is long enough, the time average of $\epsilon(t)$ is zero and, thus, the entire last term on the rhs, is also zero.
# 
# With these simplifications, we end up with,
# 
# $$
# a = \overline{x(t)x(t - \Delta t)}
# $$
# 
# This expression should look somewhat familiar. Recall the expression for **autovariance** ($\gamma$). The above expression is simply a statement of the autovariance for $\tau$ = 1 ($\Delta t$ -> $\tau$ = 1).
# 
# That is, ***$a$ is the autocorrelation at lag $\Delta t$ (aka the lag-1 autocorrelation)***! Cool!
# 
# $$
# a = \gamma(\tau = 1) = \gamma(\Delta t) = \gamma(1)
# $$
# 
# 
# Since $x$ is standardized, the above expression is also a statement of the **autocorrelation** ($\rho$) for $\tau$ = 1 ($\Delta t$ -> $\tau$ = 1).
# 
# $$
# a = \rho(\tau = 1) = \rho(\Delta t) = \rho(1)
# $$
# 
# What about $b$, the magnitude of the *white noise* term?
# 
# Since $x(t)$ and $\epsilon(t)$ both have unit variance, we can square both sides of the red noise equation and then take the time average to solve for $b$:
# 
# $$
# \begin{align}
# \overline{x(t)^2} & = a^2\overline{x(t - \Delta t)^2} + b^2\overline{\epsilon(t)^2}\\
# 1 & = a^2 + b^2\\
# b & = \sqrt{1 - a^2}
# \end{align}
# $$
# 
# ## Autocorrelation Function of a Red Noise Time Series
# 
# Now, that we have defined a red noise time series to be one the explicity includes autocorrelation, let's take a look at what this theoretical autocorrelation function looks like.
# 
# We will start by writing down the values of $x(t)$, $x(t + \Delta t)$ and $x(t + 2\Delta t)$, so that we can start to see a pattern emerge:
# 
# $$
# \begin{align}
# x(t) & = ax(t - \Delta t) + b\epsilon(t)\\
# x(t + \Delta t) & = ax(t) + b\epsilon(t + \Delta t)\\
# x(t + 2\Delta t) & = ax(t - \Delta t) + b\epsilon(t + 2\Delta t)\\
# \end{align}
# $$
# 
# Now, we will develop an expression for the autocorreltion function by focusing on the last example,
# 
# $$
# x(t + 2\Delta t) = ax(t - \Delta t) + b\epsilon(t + 2\Delta t)
# $$
# 
# We can derive an expression for the autocorrelation at lag $2\Delta t$ (the lag-2 autocorrelation) by multiplying both sides of the equation by $x(t)$ and taking the time average:
# 
# $$
# \overline{x(t)x(t + 2\Delta t)} = a\overline{x(t)x(t - \Delta t)} + b\overline{x(t)\epsilon(t + 2\Delta t)}
# $$
# 
# As before the time average of the last term on the rhs is zero. Thus, we can rewrite the above in terms of autocorrelation:
# 
# $$
# \begin{align}
# \rho(2\Delta t) & = a\rho(\Delta t)\\
# & = \rho^2(\Delta t)
# \end{align}
# $$
# 
# where $a = \rho(\Delta t)$ (see above).
# 
# What does this tell us? **This is important!** The above tells us that the lag-2 autocorrelation of a red-noise time series is equal to the lag-1 autocorrelation squared.
# 
# Now, we can generalize this statement by recognizing the pattern we noted above when we wrote down the red noise equation for consecutive time points. Thus, for lag-$n$, where $n$ = 1,...,$N$, the autocorrelation becomes,
# 
# $$
# \begin{align}
# \rho(n\Delta t) & = a\rho((n-1)\Delta t)\\
# & = \rho^n(\Delta t)
# \end{align}
# $$
# 
# This the **autocorrelation function for our red noise time series!**
# 
# ### Properies of the Red Noise Autocorrelation Function (ACF)
# 
# A function that behaves in the same way as the Red Noise ACF is the exponential: $e^{(nx)}$ = $(e^x)^n$. So, it turns out that the ACF for a red noise time series is an exponential:
# 
# $$
# \rho(n\Delta t) = e^{\frac{-n\Delta t}{T_e}}
# $$
# 
# where $T_e$ is is the **$e$-folding time-scale** of the autocorrelation function. 
# 
# In other words, the autocorrelation function of red noise decays exponentially as a function of lag $\tau$ = $n\Delta t$.
# 
# The $e$-folding time-scale is the time it takes for the autocorrelation to drop to 1/$e$ = 0.368 of the original value ($\rho$(0)=1), and can be computed as,
# 
# $$
# T_e = -\frac{\Delta t}{ln(a)}
# $$
# 
# Let's take a look at an example of what we mean by the $e$-folding time-scale.
# 
# ```{admonition} Question #1
# :class: tip
# Suppose we have a red noise time series with a temporal resolution of $\Delta t$ = 1 day. The lag-1 autocorrelation of the time series is $\rho(1)$ = 0.6, what is the e-folding time-scale?
# ```
# We can simply plug these values into the equation for $T_e$ above.

# In[1]:


import numpy as np

# calculate the e-folding time scale
Te = -1/np.log(0.6)
print(np.round(Te,2))


# The $e$-folding time-scale of the autocorrelation function is approximately 2 days. In other words, the time series loses approximately ~37\% of its memory of its of its previous state after ~2 days.
# 
# ## White Noise
# 
# Recall that a red noise process is defined as,
# 
# $$
# x(t) = ax(t - \Delta t) + b\epsilon(t)
# $$
# 
# A **white noise** process is a special case, where $\rho(\Delta t >$ 0) = 0, i.e. $a$ = 0
# 
# White noise has zero autocorrelation (no memory of the previous time steps). In geophysical applications, white noise is generally assumed to be normally distributed.
# 
# In terms of our previous regression example, our goal is for the ACF of our residuals to look like the ACF for white noise.
# 
# In the next section, we will take a closer look at the implications of autocorrelation in time series.

# In[ ]:




