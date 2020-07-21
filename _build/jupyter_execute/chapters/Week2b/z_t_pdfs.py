# Comparing the Z- and t-distributions

Here we compare the Z- and t-distributions for different sample sizes.

#  Load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#  Plot Z versus t distributions

dof = 2 #degrees of freedom (N-1)
x = np.arange(-6,6,.01)

#  Get distributions of z and t
z = stats.norm.pdf(x,0,1)
t = stats.t.pdf(x,dof)

#  Plot z and t
plt.figure(figsize=(10,6))
plt.plot(x,z, color = 'k', label = 'Z')
plt.plot(x,t,linestyle = '-', color = 'r', label = r"t ($\nu$ =" + str(dof) + ")")

#  Make the plot look nice!
plt.title('Z and Students t probability density functions')
plt.ylabel('f(Z)')
plt.legend(frameon = 0,fontsize = 16)
plt.xlim(-5,5)
plt.ylim(0,.45)
plt.yticks(np.arange(0,.5,.1))
plt.axhline(0)

