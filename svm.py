import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation
# Replace 'path_to_dataset.csv' with the actual path to your dataset file.
# If you encounter a UnicodeDecodeError, try specifying the appropriate encoding.
data = pd.read_csv('/content/drive/MyDrive/project/svm.csv', encoding='latin1')

# Separate features (essays) and target (scores)
Essay = data['Essay']
Score = data['Score']

# Step 2: Feature Extraction using TF-IDF
# Create a TfidfVectorizer to convert essays to numerical feature vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed

# Fit and transform the essays to obtain the TF-IDF feature vectors
feature_vectors = tfidf_vectorizer.fit_transform(Essay)

# Step 3: Split Data into Training and Test Sets
# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, Score, test_size=0.2, random_state=42)

# Now you can proceed with the SVM model training using the training data (X_train and y_train).
# After training the model, you can evaluate its performance on the test data (X_test and y_test).
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 4: Model Training
# Create an SVM model
svm_model = SVR(kernel='linear')  # You can use different kernels like 'linear', 'rbf', etc.
# Train the SVM model using the training data
svm_model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Use the trained SVM model to predict scores for the test data
y_pred = svm_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mse ** 0.5
# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
print(f"R-mse (RMSE) Score: {rmse:.2f}")
# Step 6: Apply the Model for Automated Essay Scoring (with user input)
# Get user input for a new essay
# Read the new essay from a text file
file_path = '/content/drive/MyDrive/project/essay/essay1.txt'  # Path to the text file containing the new essay
with open(file_path, 'r') as file:
    new_essay = file.read()

# Convert the user's essay into a list (as we'll use the same code for multiple essays)
new_essays = [new_essay]

# Convert new essay into feature vectors using the same TF-IDF vectorizer
new_feature_vectors = tfidf_vectorizer.transform(new_essays)

# Use the trained SVM model to predict scores for the new essay
predicted_scores = svm_model.predict(new_feature_vectors)

# Print the predicted score for the new essay
print(f"Predicted Score: {predicted_scores[0]}")

