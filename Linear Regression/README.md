# Linear Regression

**Linear Regression** is one of the simplest machine learning algorithm that we use. It can be described as :

- It is supervised machine learning algorithm which main aim is to find a line that minimizes the prediction error of all data points.
- Linear approach to modeling the relationship between dependent variable(Y) and one or more independent variables(X).
- When we fit a line to data using linear regression, we optimize the **slope and intercept**. Linear Regression predicts continuous numerical values based on the given data points.
- Linear Regression is a linear approach to modeling the relationship between **dependent variable and one or more independent variables**
  - dependent variable = **y**
  - independent variable = **X**

### We define a linear relationship  between x and y by using the equation **Y = mX + b** , where m represents **slope** and b represents **y-intercept**

### Loss Function
The loss function is also know as 'cost Function', i.e. the error in the predicted value of m and b, which is given by **Y = mX + b**

### Mean Squared Error(MSE)
In Statistics, Mean Square Error (MSE) is defined as Mean or Average of the square of the difference between actual and estimated values. This is also used as a measure for model evaluation

MSE is used to check how close estimates or forecasts are to actual values. Lower the MSE, the closer is forecast to actual. This is used as a model evaluation measure for regression models and the lower value indicates a better fit.

### Gradient Descent 
Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.

# Pros/Cons Multiple Linear Regression

### Pros
- Easy to implement, theory is not complex, low computational power compared to other algorithms.
- Perfect for linearly seperable datasets.
- Easy to interpret coefficients for analysis.

### Cons
- Unlikely in the real world to have perfectly linearly separable datasets, model often underfits in real-word scenarios or is outperformed by other ML and Deep Learning algorithms.
