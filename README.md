# P138002_Assignment3_DataManagement
# Spark Machine Learning For Iris Dataset

![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/ee751cc1-ba52-43d9-94cb-38a09f9e71f9)


## Overview and Introduction
In this assignment, Spark MLlib will be utilized to perform classification on the renowned Iris dataset. The Iris dataset, a staple in the field of machine learning, is frequently used as a benchmark for testing and comparing classification algorithms. This dataset consists of 150 samples of iris flowers, each characterized by four features: sepal length, sepal width, petal length, and petal width. The task is to classify each sample into one of three species: Iris-setosa, Iris-versicolor, or Iris-virginica.

## Classification - Decision Tree
In the assignment, the Decision Tree algorithm was selected from Spark MLlib for several reasons. Decision Trees are a widely-used classification algorithm that offers several advantages, making them a suitable choice for this task.

- Decision Trees are highly interpretable, as the model can be visualized as a tree-like structure where decisions are made based on feature values. This makes it easier to understand and interpret how the model is making its predictions, which is particularly useful for educational purposes and for explaining the model to non-technical stakeholders.
- The Iris dataset contains numerical features (sepal length, sepal width, petal length, petal width). Decision Trees handle numerical data efficiently by creating splits based on the values of these features, making them an excellent choice for this dataset.
- The Iris dataset, with 150 samples, is relatively small. Decision Trees are efficient and fast to train on small to medium-sized datasets, making them an appropriate choice for this assignment.


## Coding
### Load the Iris Dataset into a Spark DataFrame

```python
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark Context
conf = SparkConf().setAppName("Iris Classification")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Load the Iris dataset from HDFS
iris_path = "/user/maria_dev/nazmi/iris2.csv"
iris_df = sqlContext.read.csv(iris_path, header=True, inferSchema=True)

# Print top 10 rows
print("Top 10 rows of iris:")
iris_df.show(10)
print(iris_df.columns)
```
![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/0d23a702-63de-4ed5-a311-0b00a9ea6884)

### Convert String Labels to Numeric Using StringIndexer, Splitting dataset, Assemble features for Classifier
```python
# Convert string labels to numeric using StringIndexer
indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
indexer_model = indexer.fit(iris_df)
indexed_df = indexer_model.transform(iris_df)

# Split the data
(training_data, testing_data) = indexed_df.randomSplit([0.7, 0.3], seed=42)

# Assemble features into a single vector
feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
```

### Initialize Decision Tree Classifier, Create Grid Search and Cross-validation
```python
dt_classifier = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")

# Create a pipeline with assembler and classifier
pipeline = Pipeline(stages=[assembler, dt_classifier])

# Create a ParamGrid for Grid Search
paramGrid = (ParamGridBuilder()
             .addGrid(dt_classifier.maxDepth, [2, 4, 6, 8])
             .addGrid(dt_classifier.impurity, ["gini", "entropy"])
             .build())
             
# Create CrossValidator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross-validation, and choose the best set of parameters
cv_model = crossval.fit(training_data)
```

### Get best model from Cross-validation
```python
best_model = cv_model.bestModel

# Get best model's parameters
best_maxDepth = best_model.stages[-1].getOrDefault('maxDepth')
best_impurity = best_model.stages[-1].getOrDefault('impurity')

# Print best parameters
print("Best Model Hyperparameters:")
print("- maxDepth: ", best_maxDepth)
print("- impurity: ", best_impurity)
```
![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/72bd7658-548e-4dbb-b213-49f1e91d0af4)

### Insights
- maxDepth: 2. The maxDepth parameter defines the maximum depth of the decision tree.
- Explanation: A depth of 2 means the tree will have at most two levels of decision nodes between the root (starting node) and the leaf nodes (final nodes). In simpler terms, the decision tree will make at most two splits to classify an instance.
Implications: A shallower tree is often less complex and is less likely to overfit the training data. Overfitting occurs when the model captures noise in the training data rather than the underlying pattern, which can lead to poor performance on new, unseen data. In this case, the optimal depth of 2 suggests that a simpler model was better at generalizing from the training data to the test data.

- impurity: "gini". The impurity parameter specifies the function used to measure the quality of a split at each node of the tree.
- Explanation: The "gini" impurity measure calculates the probability of a randomly chosen element being incorrectly classified if it was randomly labeled according to the distribution of labels in the subset. It is calculated as is the probability of an element being classified into a particular class.
- Implications: Choosing "gini" as the impurity measure means that the decision tree will make splits that try to maximize the purity of the nodes. The "gini" index is commonly used because it is efficient to compute and effective at finding good splits. In this case, the use of "gini" suggests that it provided the best balance between computational efficiency and classification accuracy for this particular dataset.

### Make predictions and evaluation of model
```python
predictions = cv_model.transform(testing_data)

# Evaluate the model
accuracy = evaluator.evaluate(predictions)
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("\nMetrics:")
print("Accuracy: %g" % accuracy)
print("Precision: %g" % precision)
print("Recall: %g" % recall)
print("F1-Score: %g" % f1)
```
![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/6e71ed73-d79a-4e67-bea8-9d5970031d2a)

### Output
```python
# Convert numeric predictions back to species names
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedSpecies", labels=indexer_model.labels)
predictions = labelConverter.transform(predictions)

# Show top 20 predictions with species names and individual features
predictions.select(*(feature_columns + ["indexedLabel", "predictedSpecies"])).show(20)
```
![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/a583b417-1923-4ffb-8737-e1f2de873df6)

### Insights
- Accuracy
- Accuracy serves as an overall measure of the model's correctness in classifying instances. With an accuracy score of 97.22%, it indicates that the decision tree correctly predicted the species for 97.22% of the instances in the test dataset. This high accuracy suggests that the model is robust and effective in distinguishing between different species of iris flowers based on their sepal and petal dimensions.

- Precision
- Precision measures the proportion of true positive predictions out of all positive predictions made by the model. At 97.40%, the precision score highlights the model's ability to correctly identify instances of a specific iris species when it predicts them as such. This metric is crucial in scenarios where correctly identifying positive instances is paramount, minimizing false positives.

- Recall
- Recall (or sensitivity) assesses the model's capability to correctly identify all positive instances out of the actual positives in the dataset. Achieving a recall score of 97.22% indicates that the model effectively captured a large proportion of actual instances of each iris species. High recall is particularly important when missing positive instances could lead to significant consequences.

- F1-score
- F1-score provides a balanced measure by considering both precision and recall. With an F1-score of 97.20%, the decision tree model demonstrates a harmonious blend of precision and recall. This score is pivotal in applications where achieving a balance between minimizing false positives and false negatives is crucial for accurate predictions.


### Confusion Matrix
```python
# Extract true labels and predicted labels
y_true = predictions.select("indexedLabel").rdd.flatMap(lambda x: x).collect()
y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# Create confusion matrix
conf_matrix = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted'))

# Rename columns and index to species names
label_to_species = {i: label for i, label in enumerate(indexer_model.labels)}
conf_matrix.columns = [label_to_species[c] for c in conf_matrix.columns]
conf_matrix.index = [label_to_species[i] for i in conf_matrix.index]

print("\nConfusion Matrix:")
print(conf_matrix)

# Stop Spark Context
sc.stop()
```
![image](https://github.com/deeagjin/P138002_Assignment3_DataManagement/assets/152348898/8417262f-4045-48c8-8df7-f505302201cc)

### Insights
The confusion matrix above summarizes the performance of a machine learning model trained to classify Iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. Each row represents the instances of a true (actual) class, while each column represents the instances predicted by the model for that class.

In this specific confusion matrix:

- Iris-setosa was predicted correctly for all instances (10 out of 10). There were no instances where Iris-setosa was incorrectly classified as another species.

- Iris-versicolor was also predicted correctly for all instances (15 out of 15). Similar to Iris-setosa, there were no misclassifications of Iris-versicolor as another species.

- Iris-virginica shows a slightly different pattern. Out of 11 instances, 10 were correctly classified as Iris-virginica. However, there was 1 instance where Iris-virginica was misclassified as Iris-versicolor.
