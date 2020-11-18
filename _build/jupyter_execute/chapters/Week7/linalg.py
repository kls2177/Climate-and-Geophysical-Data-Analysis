#!/usr/bin/env python
# coding: utf-8

# # Review of Linear Algebra
# 
# Now that we will be working with multi-dimensional data, so we need to introduce matrix notation and several linear algebra concepts. The notation and concepts that are covered below are building towards the construction of the **covariance matrix**.
# 
# ## Matrix Notation
# 
# A matrix **A** with $M$ rows and $N$ columns (dimensions $M x N$) is ordered in the following way:
# 
# $$
# \bf{A} = 
# \begin{bmatrix}
# a_{11} & a_{12} & a_{13} & ... & a_{1n}\\
# a_{21} & a_{22} & a_{23} & ... & a_{2n}\\
# . & . & . & ... & .\\
# . & . & . & ... & .\\
# a_{m1} & a_{m2} & a_{m3} & ... & a_{mn}
# \end{bmatrix}
# $$
# 
# The notation $a_{ij}$ will be used to denote the element in the $i$th row and the $j$th column, where 1 $\leq i \leq M$ and 1 $\leq j \leq N$.
# 
# ## Transpose of a Matrix
# 
# The transpose of a matrix **X** is written as **X$^T$**, and is calculated by flipping the rows and columns of the the matrix. So, if the dimensions of **X** are $M x N$, then the dimensions of **X$^T$** are $N x M$. 
# 
# Written in index notation:
# 
# $$
# \begin{align}
# \bf{X} &= \bf{Y}^T\\
# \bf{X}_{ij} &= \bf{Y}_{ij}
# \end{align}
# $$
# 
# Let’s write down an example: What is the transpose of **X**? 
# 
# $$
# \bf{X} = 
# \begin{bmatrix}
# 1 & 2 & 3 \\
# -1 & -1 & -3
# \end{bmatrix}
# $$
# 
# ```{toggle}
# $$
# \bf{X}^T = 
# \begin{bmatrix}
# 1 & -1\\
# 2 & -1\\
# 3 & -3
# \end{bmatrix}
# $$
# ```
# 
# ## Multiplying Matrices
# 
# If **A** is $M x N$ and **B** is $N x P$, then, their product can be written as,
# 
# $$
# \bf{AB} = \bf{C}\\
# $$
# 
# and the corresponding sizes of the matrices are,
# 
# $$
# [M x N] \cdot [N x P] = [M x P]
# $$
# 
# Unlike regular multiplication, one cannot re-order matrix multiplication, e.g. **AB** is ok but **BA** is not. The number of columns of the first matrix must be equal to the number of rows of the second matrix.
# 
# This is the case because, when we multiply matrices together, we multiply the row of the first matrix with the column of the second matrix and sum the values to get the value that goes in the first row and first column of the solution. Let's look at an example:
# 
# $$
# \begin{bmatrix}
# 1 & 2 & 3 \\
# -1 & -1 & -3
# \end{bmatrix} \cdot
# \begin{bmatrix}
# 2 & 3 \\
# 4 & 2\\
# -1 & 1
# \end{bmatrix}
# $$
# 
# ```{toggle}
# $$ 
# = \begin{bmatrix}
# 7 & 10 \\
# -3 & -8
# \end{bmatrix}
# $$
# ```
# 
# where the corresponding sizes of the matrices are,
# 
# $$
# [2 x 3] \cdot [3 x 2]  = [2 x 2]
# $$
# 
# 
# ## Multiplying Vectors
# 
# 
# Suppose that we have two $M x 1$ vectors, **X** and **Y**. How can we multiply these together, following the rules of matrix multiplication? There are two ways that give very different results:
# 
# $$
# \mathbf{XY}^T = \mathbf{Z}
# $$
# 
# where the corresponding sizes of the vectors/matrices are,
# 
# $$
# [M x 1] \cdot [1 x M]  = [M x M]
# $$
# 
# and 
# 
# $$
# \mathbf{X}^T\mathbf{Y} = \mathbf{W}
# $$
# 
# where the corresponding sizes of the vectors/matrices are,
# 
# $$
# [1 x M] \cdot [M x 1]  = [1 x 1]
# $$
# 
# ## Inner Product of Two Vectors
# 
# The second example of multiplying vectors is called the **inner product** of two vectors. This is important to keep in mind as we build toward constructing the **covariance matrix**. Before we get to the covariance matrix; however, we will look at how we can construct the *variance* of a vector **X** by computing its inner product. To relate what we will do below, to what we have already covered in the courseware, we can think of a vector as a time series. 
# 
# Let's show that if **X** is a vector with mean removed, then **X$^T$X** is the same as the variance of ($M$-1)**X**. **X$^T$X** is known as the inner product of **X**:
# 
# $$
# \begin{align}
# \mathbf{X^TX} &= \sum_{n=1}^{M} x_i^{\prime}x_i^{\prime}\\
# &= \sum_{n=1}^{M} x_i^{\prime2}\\
# & = \text{scalar quantity}
# \end{align}
# $$
# 
# This should look very much like the equation for the [*sample variance*](https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis/chapters/Week1b/summarystats.html) without the factor of $\frac{1}{M-1}$ in front.
# 
# The inner product of two $M x 1$ vectors, **X** and **Y**, represents the covariance between **X** and **Y**,
# 
# $$
# \begin{align}
# \mathbf{X^TY} &= \sum_{n=1}^{M} x_i^{\prime}y_i^{\prime}\\
# & = \text{scalar quantity}
# \end{align}
# $$
# 
# 
# 
# ## The Covariance Matrix
# 
# Now, suppose you have a matrix **X** of dimensions $M x N$. If the columns of **X** have mean zero (you can think of each column as an anomaly time series), then
# 
# $$
# \mathbf{C} = \frac{1}{M-1} \mathbf{X^TX}
# $$
# 
# where **C** is the **covariance matrix** of dimensions $N x N$. If the columns of **X** are standardized, then **C** is the *correlation matrix*.
# 
# Let's take a closer look at what the covariance matrix is using a 2$x$2 matrix where each colunm has a mean of zero:
# 
# $$
# \mathbf{X} = \begin{bmatrix}
# x_{11}^{\prime} & x_{12}^{\prime} \\
# x_{21}^{\prime} & x_{22}^{\prime}
# \end{bmatrix}
# $$
# 
# The covariance matrix for **X** is then,
# 
# $$
# \begin{align}
# \frac{1}{M-1} \mathbf{X^TX} &= \begin{bmatrix}
# x_{11}^{\prime} & x_{21}^{\prime} \\
# x_{12}^{\prime} & x_{22}^{\prime}
# \end{bmatrix} \cdot
# \begin{bmatrix}
# x_{11}^{\prime} & x_{12}^{\prime} \\
# x_{21}^{\prime} & x_{22}^{\prime}
# \end{bmatrix}\\
# & = \frac{1}{M-1}\begin{bmatrix}
# x_{11}^{\prime2}+x_{21}^{\prime2} & x_{11}^{\prime}x_{12}^{\prime} + x_{21}^{\prime}x_{22}^{\prime} \\
# x_{11}^{\prime}x_{12}^{\prime} + x_{21}^{\prime}x_{22}^{\prime} & x_{12}^{\prime2} + x_{22}^{\prime2}
# \end{bmatrix}
# \end{align}
# $$
# 
# This looks pretty messy, but we can start to understand what the different elements of this matrix are by again assuming that each column of **X** represents an anomaly time series. We can compute the variance for each column in the usual way:
# 
# #### Column 1:
# 
# $$
# \begin{align}
# \text{Variance of Column 1} &= \frac{1}{M-1}\sum_{n=1}^{M} x_{i1}^{\prime2}\\
# &= \frac{1}{M-1}\left(x_{11}^{\prime2} + x_{21}^{\prime2}\right)
# \end{align}
# $$
# 
# #### Column 2:
# 
# $$
# \begin{align}
# \text{Variance of Column 2} &= \frac{1}{M-1}\sum_{n=1}^{M} x_{i2}^{\prime2}\\
# &= \frac{1}{M-1}\left(x_{12}^{\prime2} + x_{22}^{\prime2}\right)
# \end{align}
# $$
# 
# Note that these variances appear in the covariance matrix above along the diagnoal. The off-diagonal elements show the covariances, i.e., the *inner product* of column 1 and column 2. 
# 
# $$
# \begin{align}
# \text{Covariance of Column 1 and Column 2} &= \frac{1}{M-1}\sum_{n=1}^{M} x_{i1}^{\prime}x_{i2}^{\prime}\\
# &= \frac{1}{M-1}\left(x_{11}^{\prime}x_{12}^{\prime} + x_{21}^{\prime}x_{22}^{\prime}\right)
# \end{align}
# $$
# 
# ## Inverse of a Matrix
# 
# The inverse of a square matrix is an incredibly important and useful concept. 
# 
# The inverse of square matrix **X** is written as **X**$^{-1}$. 
# 
# In matrix notation, this does not mean raise the matrix to the -1 power, but rather, it denotes the inverse specifically. The inverse has a very special property such that,
# 
# $$
# \mathbf{XX^{-1}} = \mathbf{X^{-1}X} = \mathbf{I}
# $$
# 
# where **I** is the identity matrix, which is a square matrix with 1's along the diagonal and zeros elsewhere.
# 
# 
# Note that only square matrices have inverses. In algebra, if you have a variable multiplied on one side of an equation, you use division to get rid of it. With matrices, you use the inverse!
# 
# ## Types of Matrices
# 
# * **X** is diagonal if: **X** is square, with values along the diagonal and zeros everywhere else
# 
# * **X** is symmetric if: **X**$^T$ = **X**
# 
# * **X** is orthonormal if: **X**$^T$ = **X**$^{-1}$ or/and **X$^T$X** = **XX$^T$** = **I**
#  * i.e., if all of the off-diagonal elements of **X$^T$X** are zero, then **X**$^T$ = **X**$^{-1}$, and all the columns in **X** are linearly independent (not correlated with one another).
# 
# 
# ## Vector Spaces and Basis Vectors
# 
# A vector space (**R$^N$**) is a fancy way of describing the dimensionality of domain we are working in. Here are some examples:
# 
# * **R$^0$**: a single point (e.g. 0)
# 
# * **R$^1$**: the real line
# 
# * **R$^2$**: 2-D space
# 
# * **R$^3$**: 3-D space
# 
# Vector spaces are spanned by a set of basis vectors. The basis for **R$^N$** is a set of $N$ vectors that through linear combination can describe any vector in **R$^N$**.
# 
# While there are an infinite number of basis vectors for any **R$^N$** ($N$ > 0), the typical vectors chosen are the unit vectors. Let's work through some examples.
# 
# For **R$^2$**, there are 2 basis vectors, each of length 2 $x$ 1. The unit vectors are,
# 
# $$
# \begin{align}
# \mathbf{e_1} &= (1,0)\\
# \mathbf{e_2} &= (0,1)
# \end{align}
# $$
# 
# Any point in **R$^2$** can be written as a linear combination of $e_1$ and $e_2$. For example, the point (3,2) is can be written as
# 
# $$
# (3,2) = 3\mathbf{e_1} + 2\mathbf{e_2}
# $$
# 
# Let's look at another example. Suppose we have the following vectors:
# 
# $$
# \begin{align}
# \mathbf{e_1} &= (1,0)\\
# \mathbf{e_2} &= (2,0)
# \end{align}
# $$
# 
# Do these form a basis for **R$^2$**? 
# 
# The answer is “no”, since any point with a non-zero second index cannot be obtained through linear combination. In this case, the space (0,1) and all linear combinations of this vector is called the *null space* since it cannot be reached by the vectors listed ($e_1$, $e_2$). More on null spaces later.
# 
# How about for **R$^3$**? The basis vectors in this case are 3 vectors each of length 3 $x$ 1
# 
# $$
# \begin{align}
# \mathbf{e_1} &= (1,0,0)\\
# \mathbf{e_2} &= (0,1,0)\\
# \mathbf{e_3} &= (0,0,1)
# \end{align}
# $$
# 
# By definition, basis vectors must be *linearly independent* - that is, each provides unique information that the others do not.
# 
# ### The Determinant
# 
# To check if a set of vectors form a basis, we ask if there is a vector $x$ such that
# 
# 
# $$
# \mathbf{Ax} = 0
# $$
# 
# where **A** is a matrix made of the vectors we are testing. If there is a non-trivial solution to this equation (there is a unique, non-zero vector $x$), then the vectors do not span the entire space and/or are not all independent. In other words, If $x$ exists, then the vectors do not form a basis and there is a non-trivial null-space of **A**.
# 
# Instead of solving the above equation for specific $x$'s, we really just want to know if a solution exists or not. Turns out, there is a shortcut. If
# 
# $$
# \text{det}(\mathbf{A}) = \begin{vmatrix}\mathbf{A}\end{vmatrix} = 0
# $$
# 
# then there is a non-trivial solution $x$ to
# 
# $$
# \mathbf{Ax} = 0
# $$
# 
# where det(**A**) is the **determinant** of **A**.
# 
# We’re not going to go into the details of how to calculate determinants here, because it’s quite cumbersome, but, as an example, here is the determinant for a 2 $x$ 2 matrix:
# 
# $$
# \text{det}(\mathbf{A}) = \begin{vmatrix}\mathbf{A}\end{vmatrix} = a_{11}a_{22} - a_{12}a_{21}
# $$
# 
# So, if the matrix **A** is square, and there is a non-trivial solution to
# 
# $$
# \mathbf{Ax} = 0
# $$
# 
# then **A** is termed *singular* or *degenerate* and does not have an inverse.
# 
# 
# ## Rank of a Matrix and Linear Independence
# 
# The rank $r$ of a matrix is the number of linearly independent columns. Returning to our previous example, if **A** holds the basis vectors for **R^2**, then
# 
# $$
# \mathbf{A} = \begin{bmatrix}
# 1 & 0 \\
# 0 & 1
# \end{bmatrix}
# $$
# 
# We can check that the columns of **A** are linearly independent by confirming that
# 
# $$
# \text{det}(\mathbf{A}) = 1 \neq 0
# $$
# 
# Thus, the rank of **A** is 2.
# 
# However, if
# 
# $$
# \mathbf{A} = \begin{bmatrix}
# 1 & 2 \\
# 0 & 0
# \end{bmatrix}
# $$
# 
# Then
# 
# $$
# \text{det}(\mathbf{A}) = 0
# $$
# 
# and we see the columns are not linearly independent. In this instance, the rank $r$ of **A** is 1, even though **A** is 2 $x$ 2.

# In[ ]:





# In[ ]:




