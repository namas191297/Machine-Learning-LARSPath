import numpy as np
import matplotlib.pyplot as plt
from sklearn import *

# Importing data from the datasets existing in the scikit-learn datasets.
wines = datasets.load_wine()

# Dividing data into predictor/independent variables and dependent variables(residual)
X = wines.data
y = wines.target

# As the data above are matrices, we can print the shape of them. ie. [rows,columns]
print("Independent variables: {} rows and {} columns".format(X.shape[0],X.shape[1]))
print("Dependent variable: {} rows".format(y.shape[0]))

#Determining different parameters involved in the model.
alphas, indicesNotZero, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

#Displaying the parameter values.
print("The values of the lambda used in the penalty applied during LASSO(L1-norm): {}".format(alphas))
print("LASSO induces sparsity, which means less non-zero coefficients, hence, the indices of coefficients with nonzero value are: {}".format(indicesNotZero))
print("The optimal co-efficient values estimated for every different value of lamda are: {}".format(coefs))


# Preparing data for visualization of the regularization path.
sumofcoefficients = np.sum(np.abs(coefs.T), axis=1)         # The absolute sum of all the co-efficients at each model/lamda-value as done in the L2 norm. (|sigma(lambda)|)

print(alphas[0])
#Printing the data
print("The sum of co-efficients for each lambda-value are as follows:")
for i in range(len(sumofcoefficients)):
    print("Lambda value: {}, Sum of coefficients: {}".format(alphas[i],sumofcoefficients[i]))

#Other calculations
sumofcoefficients /= sumofcoefficients[-1]     # Divides all the sums by the highest value in the array to normalize the range to 0-1.
print("The normalized sums of co-efficients are: {}".format(sumofcoefficients))


#Plotting the graph to visualize the regularization path
plt.plot(sumofcoefficients, coefs.T)
ymin, ymax = plt.ylim()                                             # Retrieving the maximum and minimum values on the y-axis
plt.vlines(sumofcoefficients, ymin, ymax, linestyle='dashed')       # Draw vertical lines to all the the sum of coefficients on the X axis.
plt.xlabel('Normalized Sum of Coefficients')
plt.ylabel('Coefficients')
plt.title('Regularization path for LASSO model')
plt.axis('tight')
plt.show()                                                          # Display the graph