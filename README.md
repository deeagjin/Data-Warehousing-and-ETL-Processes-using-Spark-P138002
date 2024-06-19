# P138002_Assignment3_DataManagement

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
