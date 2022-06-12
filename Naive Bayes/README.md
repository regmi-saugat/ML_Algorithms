# Naive Bayes 
Naive Bayes is one of the supervised machine learning algorithm which is classification technique based on Bayes’ Theorem with an assumption of independence among predictors.

Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.Naive Bayes is known to outperform even highly sophisticated classification methods.

# Pros and Cons

### Pros
- It is easy and fast to predict class of test data set. If this assumption holds true, performs exceptionally well.
- Works well with high dimensions, works well with problems such as text classification.
- It also perform well in multi class prediction.
- It perform well in case of categorical input variables compared to numerical variable.

### Cons
- It assumes all features are independent, in real life, it is almost impossible.
- If the categorical variable has a category in the test data set, which was not observed in the training data set, the model assigns a zero probability to this category and fails at making a prediction. Use smoothing to deal with this issue.
  - Smoothing is a technique in detecting trends with noisy data for cases where the shape of the trend is unknown. Laplace Smoothing is common with Naive Bayes, it is used with categorical data and meant to allevaite the problem of zero probability. Attaching an additional link in bullet below about smoothing in relation to Naive Bayes.

## Some applications of Naive Bayes Algorithms
- Real time Prediction
- Multi class Prediction
- Recommendation System
- Text classification / Spam Filtering / Sentiment Analysis
