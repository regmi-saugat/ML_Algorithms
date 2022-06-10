# Logistic Regression
Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is binary. It is used to describe data and to explain the relationship 
between one dependent binary varible and one or more nominal, ordinal, interval or ratio-level varaibles.

# Pros & Cons:

## Pros
- Simple algorithm that is easy to implement, does not require high computation power.
- Performs extremely well when the response variable is linearly seperable.
- Less prone to over - fitting with low - dimensional data.
- Very easy to interpret, can give a measure of how relevant a predictor is and the association.

## Cons
- Logistic Regression has a linear decision surface that separates its classes in its predictions, in the real world it is extremely rare that you will have linearly seperable data.
- Need to perform carefully data exploration, logistic regression suffers with datasets with high multicollinearity between variables, repetition of information can lead to wrong training of parameters.
- Algorithms is sensitive to outliers.
- Hard to capture complex relationships, deep learning and classifiers such as random forest can outperform with more realistic datasets.
