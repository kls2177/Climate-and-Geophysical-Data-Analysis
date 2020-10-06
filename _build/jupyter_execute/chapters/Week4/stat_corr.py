#!/usr/bin/env python
# coding: utf-8

# # Statistical Significance of Correlation Coefficients
# 
# ## When $\rho$ = 0:
# 
# When $\rho$ = 0, we can use a form of the $z$-statistic and $t$-statistic.
# 
# $$
# t = \frac{r - \rho}{\sqrt{\frac{1-r^2}{N-2}}}
# $$
# 
# where the standard error of the distribution of $r$ is $\sqrt{\frac{1-r^2}{N-2}}$ and $\rho$ = 0.
# 
# For large $N$, this statistic is normally distributed (like the $z$-statistic), but for small $N$ it has the distribution like a $t$-statistic, with $\nu$ = $N$ - 2 degrees of freedom. 
# 
# As before, for small $N$ the above statistic is only valid if the underlying distributions of the variables being correlated are normal. If $N$ is large, then the Central Limit Theorem applies and the underlying distributions do not need to be normal.
# 
# 
# Let's take a look at an example:
# 
# ```{admonition} Question #1
# :class: tip
# We have two time series, each of length 20, and they are correlated at $r$ = 0.6. Does this correlation exceed the 95\% confidence interval under the null hypothesis that $\rho$ = 0? You can assume both time series are sampled from underlying normal distributions.
# ```
# 
# We had no prior knowledge (before getting the samples) that the sample correlation would be positive or negative, so we will use a two-sided $t$-test. The critial $t$-value for the 95\% confidence level when dof = $N$ - 2 = 18 is:

# In[1]:


# critical t-value
import numpy as np
import scipy.stats as st
N = 20
dof = N - 2
t_crit = st.t.ppf(0.975,dof)
print(np.round(t_crit,2))


# So, how does our actual $t$-statistic compare?

# In[2]:


# calculate the t-statistic
r = 0.6

t = r * np.sqrt(N-2)/np.sqrt(1-r**2)
print(np.round(t,2))


# ##### Since the $t$-statistic is larger than $t_c$, we can reject the null hypothesis that the population correlation is zero ($\rho$ = 0).
# 
# ## When $\rho \neq$ 0:
# 
# As we saw in the last section, when the true population correlation is not zero, the sampling distribution is not symmetric (like the $z$ or $t$ distributions), and so we cannot use the properties of the normal distribution for hypothesis testing.
# 
# However, we can transform the sampling distribution of $r$ into something that is normal using the **Fisher-Z Transformation** to obtain the **Fisher-Z statistic**.
# 
# $$
# \text{Fisher-Z} = \frac{1}{2}ln\left(\frac{1+r}{1-r}\right)
# $$
# 
# The Fisher-Z statistic is normally distributed with:
# 
# $$
# \begin{align}
# \mu_Z &= \frac{1}{2}ln\left(\frac{1+\rho}{1-\rho}\right)\\
# \sigma_Z &= \frac{1}{\sqrt{N-3}} 
# \end{align}
# $$
# 
# The [table](fisherz) below shows values of $r$ and the corresponding Fisher-Z statistic.
# 
# ```{list-table}
# :header-rows: 1
# :name: fisherz
# 
# * - Correlation Coefficient
#   - Fisher-Z Statistic
# * - 0.00
#   - 0.00
# * - 0.30
#   - 0.31
# * - 0.50
#   - 0.55
# * - 0.70
#   - 0.88
# * - 0.90
#   - 1.47
# * - 0.99
#   - 2.65
# ```
# 
# ### Confidence Interval on $\rho$
# 
# To calculate the confidence interval for correlations (even $\rho$ = 0), we need to use the Fisher-Z transformation in the same way that we calculated the confidence interval using the $z$-statistic.
# 
# $$
# \begin{align}
# Z - z_c\sigma_Z \leq & \mu_Z \leq Z + z_c\sigma_Z\\
# Z_l \leq & \mu_Z \leq Z_u
# \end{align}
# $$
# 
# You can then transform the lower ($Z_l$) and upper ($Z_u$) confidence limits back to correlation using the following expression:
# 
# $$
# \begin{align}
# r_{l,u} & = \frac{e^{2Z_{l,u}}-1}{e^{2Z_{l,u}}+1}\\
# & = tanh(Z_{l,u})
# \end{align}
# $$
# 
# Let's look at our previous example and calculate the confidence interval using the Fisher-Z statistic:
# 
# ```{admonition} Question #2
# :class: tip
# We have two time series, each of length 20, and they are correlated at $r$ = 0.6. Compute the 95\% confidence interval for $\rho$. You can assume both time series are sampled from underlying normal distributions.
# ```
# We already saw that we can reject the null hypothesis that the population correlation is zero for this example. So, can we get a sense of where the true correlation lies by computing the confidence interval?
# 
# There is no python function to perform the Fisher-Z transformation, so we have to do this by hand.

# In[3]:


r = 0.6
N = 20

# calculate Fisher-Z statistic and the standard error
FZ = 0.5*np.log((1+r)/(1-r))
sigma_Z = 1/np.sqrt(N-3)
print(np.round(FZ,3),np.round(sigma_Z,3))


# Now, the we have our Fisher-Z statistic and the corresponding standard error, we can compute the critical $z$-value for the 95\% confidence interval.

# In[4]:


# critical z-value for two-sided 95% confidence interval
z_crit = st.norm.ppf(0.975)
print(np.round(z_crit,2))


# Just as we did with previously with $z$-statistic, we compute the confidence interval.

# In[5]:


# upper and lower confidence limits
Z_upper = FZ + z_crit*sigma_Z
Z_lower = FZ - z_crit*sigma_Z


# And then, we transform the confidence limits back to correlation.

# In[6]:


# transform from Z back to correlation
r_upper = np.tanh(Z_upper)
r_lower = np.tanh(Z_lower)
print(np.round(r_lower,2),np.round(r_upper,2))


# Thus, the population correlation lies between
# 
# $$
# 0.21 \leq \rho \leq 0.82
# $$
# 
# with 95\% confidence.
# 
# ### Comparing Two Non-Zero Correlations
# 
# If we want to test the difference between two correlations that are non-zero, we can once again use the Fisher-Z transformation for each and use the fact that $Z$ is normally distributed.
# 
# Suppose we have two samples of size $N_1$ and $N_2$  which give correlation coefficients of $r_1$ and $r_2$. We test for a significant difference between these correlations by first performing the Fisher-Z transformation for each:
# 
# $$
# \begin{align}
# \text{Z_1} & = \frac{1}{2}ln\left(\frac{1+r_1}{1-r_1}\right)\\
# \text{Z_2} & = \frac{1}{2}ln\left(\frac{1+r_2}{1-r_2}\right)
# \end{align}
# $$
# 
# and then calculating our normal $z$-statistic from the difference of means:
# 
# $$
# z = \frac{Z_1 - Z_2 - \Delta_{1,2}}{\sigma_{1,2}}
# $$
# 
# where
# 
# $$
# \Delta_{1,2} = \mu_1 - \mu_2
# $$
# 
# is the transformed difference you expect (your null hypothesis). If your null hypothesis is that the population correlations of the two samples are equal ($\rho_1$ = $\rho_2$), then
# 
# $$
# \Delta_{1,2} = 0
# $$
# 
# and
# 
# $$
# \begin{align}
# \sigma_{1,2} & = \frac{1}{\sqrt{\sigma_1^2 + \sigma_2^2}}\\
# & = \frac{1}{\sqrt{\frac{1}{N_1 - 3} + \frac{1}{N_2 - 3}}}
# \end{align}
# $$

# In[ ]:




