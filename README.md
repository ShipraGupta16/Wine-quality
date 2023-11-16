# Wine Quality Predictions using Machine Learning

##### Group Members: Katherine Hong, Mimi George, Shipra Gupta, Fiza Tariq, Nivetha Sundar

##### Dataset Source: UCI ML Archives - [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

### Tools Used
#### Date Model Implementation: PySpark, Pandas
#### Visualizations: Seaborn, Matplotlib
#### Machine Learning Optimization: Scikit-Learn (Random Forest Classifier, PCA clustering), Tensorflow, Joblib

### Proposal

Our goal was to utilize two separate datasets dedicated to red and white wine and analyze them against various components that make wine great, like acidity, citric acid, pH, etc. With this information, we were keen to explore,

- How does each component or ingredient relate to the quality of wine? Does it differ between red and white wine?
- What are some important factors to consider when measuring the quality of both types of wine? Are there significant differences or outliers?
- Can we group/cluster wines based on their chemical compositions? 
- How accurate can certain machine learning models be in predicting the type of wine over quality?

### Analysis

1. After zipping both datasets for red and white wine, we began by looking into the distribution of our data and we found the following: 
- Our dataset included more white wine than red wine instances
- The majority of our wines were of average quality

<img src="www.github.com/ShipraGupta16/Wine-quality/tree/main/Images/wine_count.png">

2. In our correlation plot, we dig deeper into how each ingredient correlates with the other, and how all of this contributes to the quality of wine.

- Immediately we notice that volatile acidity, chlorides, density, and alcohol content have relatively extreme relationships with quality.
- Other components like sulfur dioxide (preservatives ensuring the longevity of wine) positively correlate with residual sugars, as this means that the sugar naturally occurring in grapes is preserved longer. However, this is irrelevant when it comes to wine quality.
- It’s also observed with wine of any type, the more alcohol there is in wine, the less density it will have. And higher alcohol content means better quality.

3. We performed a pair plot to compare each attribute determining wine quality.
- The blue represents the red wine and the green represents the white wine.
- In our pair plots, we see obvious relationships between volatile acidity and alcohol.
- While alcohol is equally distributed for red and white types, red wine tends to have more acidic properties.

4. Volatile acidity has a relatively negative relationship with quality, and this is more prevalent with red wine. It does not seem to affect the quality of white wine. Both red and white wine have a positive relationship when we compare alcohol against quality. It would seem that anything under 10.0% alcohol content would lower the quality score of both red and white wine.


#### Machine Learning Optimization

1. PCA Clustering / Unsupervised Learning

Since our data was primarily a numerical dataset, we found it best to begin clustering our data given the various components in our dataset. We had 15 features that contributed to a lot of noise in our dataset. Based on our heat map and pair plots, we reduced our data frame to 4 features that heavily affect the quality of wine. By doing so, we 
see clear distinctions between each cluster that allow us to categorize how different features set apart each cluster.
achieve a cumulative variance of 78%

2. Random Forest Classifier / Supervised Learning

We began by using a Random Forest Classifier model using two classes to analyze the accuracy of predictions based on the type of wine. We found our model had very high accuracy. A [source](file:///Users/nivethasundar/Downloads/SDPIT2022-400-408.pdf) published by the University of California, Davis speaks extensively on the accuracy of the Random Forest model when training our dataset. This can confirm that there is no leakage in our dataset.

Upon this, we decided to pivot into using the same model to predict the quality of wine instead. We binned our quality into 3 classes (below average, average, and above average) and ran predictions again to find that our model had an accuracy rate of 73%.

3. Plotting Decision Tree for Wine Type

Our decision tree helped us understand which features heavily contributed to our wine-type predictions. After digging deeper, we notice that sulfur dioxide, chlorides, and volatile acidity play a major role in predicting the type of wine.
We also plotted a 3D scatter plot to show how these features relate to each other depending on the wine type 0 for low quality and 1 for high quality wines.

4. Neural Network Model / Deep Learning

Attempt 1 - Dropped the “type” column and utilized all features present in our original data set.
Began with 3 hidden layers and 1 input as a start.
Attempt 2 (OPTIMIZED MODEL) - We didn’t change our data frame but instead increased our hidden layers and the amount of nodes in each layer.
Attempt 3 - We retained columns that were seen to be important features contributing to wine quality predictions in our Random Forest Classifier model. We also reduced our hidden layers once more and used a “tanh” activation function.

#### Conclusion

- Our Neural Network models and optimization yielded higher accuracy when predicting quality, specifically, attempt 2 with an accuracy rating of ~81%
- PCA clustering results in distinct clusters and a healthy variance ratio
- Strong relationships were identified in our correlation maps and pair plots.
    - For red wine
        - strong positive correlation with quality - alcohol content, sulfates, citric acid.
        - strong negative correlation with quality - volatile acidity, total sulfur dioxide.
    - For white wine
        - strong positive correlation with quality - alcohol
        - strong negative correlation with quality - density, volatile acidity, and chlorides.
