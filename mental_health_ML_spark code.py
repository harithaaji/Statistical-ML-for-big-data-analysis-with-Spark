# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("MentalHealthClassification").getOrCreate()

# Load data into Spark DataFrame
data = spark.read.csv("survey.csv", header=True, inferSchema=True)

# Display initial data overview
data.show(5)

# Select and preprocess relevant columns
feature_columns = [
    "Timestamp", "Age", "Gender", "Country", "state", "self_employed", "family_history", "treatment",
    "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor",
    "mental_health_interview", "phys_health_interview", "mental_vs_physical", "obs_consequence", "comments"
]

# Filter the data to only include the relevant columns
data = data.select(feature_columns)

# Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numeric using StringIndexer
categorical_columns = [col for col in data.columns if data.schema[col].dataType == 'string']
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed").fit(data) for col in categorical_columns]

# Vectorize the features (convert features into a feature vector)
feature_columns = [col + "_indexed" for col in categorical_columns] + ["Age"]  # Add other numerical features if needed
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Index the target column 'treatment' (binary: Yes/No)
indexer = StringIndexer(inputCol="treatment", outputCol="label")
data = indexer.fit(data).transform(data)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(labelCol="label", featuresCol="features"),
    "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features"),
    "GradientBoosting": GBTClassifier(labelCol="label", featuresCol="features"),
    "SVM": LinearSVC(labelCol="label", featuresCol="features"),
    "MLP": MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=[len(feature_columns), 10, 2]),
}

# Set up the evaluator for model performance
evaluator = BinaryClassificationEvaluator(labelCol="label")

# Evaluation setup
metrics_data = []

# Training and Evaluation
for name, model in models.items():
    print(f"\nTraining Model: {name}")
    
    # Define the pipeline
    pipeline = Pipeline(stages=indexers + [assembler, model])
    
    # Fit the model on training data
    model_trained = pipeline.fit(train_data)
    
    # Predictions on test data
    predictions = model_trained.transform(test_data)
    
    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)
    metrics_data.append((name, accuracy))
    print(f"{name} - Accuracy: {accuracy:.4f}")
    
# Convert metrics data to Pandas DataFrame for visualization
metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Accuracy"])

# Visualize accuracy
metrics_df.plot(x='Model', kind='bar', figsize=(10, 5), title='Model Performance Metrics', y=["Accuracy"])
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# If needed, you can also visualize the ROC curve using matplotlib.
# You can get ROC data from the BinaryClassificationEvaluator or other metrics as needed.
