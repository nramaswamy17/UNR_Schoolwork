import discriminant
import data_generation
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


## Problem 1

# Define Parameters
n1, n2 = 10, 10 # sample sizes
mean1, mean2 = [1,1], [4,4]
cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]

# discriminant.experiment_1(n1, n2, mean1, mean2, cov1, cov2)

## Problem 2

# Define Parameters
n1, n2 = 5, 5 # sample sizes
mean1, mean2 = [1,1], [4,4]
cov1, cov2 = [[1,0],[0,1]], [[4,0],[0,8]]
data_A1 = np.random.multivariate_normal(mean1, cov1, n1)
data_A2 = np.random.multivariate_normal(mean2, cov2, n2)

discriminant.experiment_2(data_A1, data_A2, n1, n2, cov1, cov2)
discriminant.experiment_3(data_A1, data_A2, n1, n2, cov1, cov2)

## Problem 3

# Define Parameters
n1, n2 = 60000, 140000 # sample sizes
mean1, mean2 = [1,1], [4,4]
cov1, cov2 = [[1,0],[0,1]], [[1,0],[0,1]]
data_A1 = np.random.multivariate_normal(mean1, cov1, n1)
data_A2 = np.random.multivariate_normal(mean2, cov2, n2)

#discriminant.experiment_1pt2(data_A1, data_A2, n1, n2, cov1, cov2)
#discriminant.experiment_3(data_A1, data_A2, n1, n2, cov1, cov2)
