LARS (Least Angle Regression) can be broadly defined as a Stage-wise regression made fast and efficient.
It is used to update the weights optimally by taking optimal leaps/changes in optimal directions,
while maintaining equiangularity between the residual(Y) and the variables instead of the gradual and
inconsistent updates to the weights/coefficients. It is generally used to select the variables based
on their correlation with the dependent variables. An optimal increase in the coefficients/weights is
done of the most correlated variable at a particular instance instead of fully adding the variable. 
This remedies the greedyness of the stepwise selection method and also takes optimal leaps to fit the
model efficiently. Hence, this technique is used to select independent variables while maintaining correlation between
these independent variables and the residual(independent variable). The regularization path is nothing but a
a vector of points p. P can be defined as point in n-dimensional space, where each point is a vector of coefficients
that are estimated for a particular value of lambda.

Problem: We use the wine dataset to plot the regularization path of the problem. We compute the values of alphas(lambdas), the
sparse indices (indices of variables that are non-zero) and the co-efficients that are estimated.
We display all the values of the parameters and compute the sum of co-efficients after applying a transpose operation on it, 
so we can find the sum vector or co-efficients, where each element of the vector(each sum) is the sum of all the elements of the coefficients
matrix at the same indices.
We normalize the sum vector to range from 0-1 and then plot the regularization path where X = Coefficients.Transpose and
Y = Normalized Sum of Coefficients.