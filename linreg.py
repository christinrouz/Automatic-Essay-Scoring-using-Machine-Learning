import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the dataset from CSV file
df = pd.read_csv('/content/drive/MyDrive/project/lre.csv', encoding='latin1')

# Check the column names in the dataset
#print(df.columns)

# Clean the essay text
def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', str(text).lower())
    return cleaned_text

essay_column = 'Essay'  # Update with the correct column name
if essay_column in df.columns:
    df['cleaned_essay'] = df[essay_column].apply(clean_text)
else:
    print(f"Column '{essay_column}' does not exist in the dataset.")

# Tokenize the essays
nltk.download('punkt')
df['tokenized_essay'] = df['cleaned_essay'].apply(word_tokenize)

# Print the preprocessed essays
for index, row in df.iterrows():
    essay_id = row['essay_id']
    tokenized_essay = row['tokenized_essay']
    #print(f"Essay_id: {essay_id}")
    #print(tokenized_essay)
    #print()

# Perform feature extraction using CountVectorizer
text_data = df['cleaned_essay'].astype(str)
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(text_data)

# Convert the extracted features into a DataFrame
feature_names = vectorizer.get_feature_names_out()
feature_df = pd.DataFrame(features.toarray(), columns=feature_names)

# Concatenate the feature DataFrame with the original data
extracted_data = pd.concat([df, feature_df], axis=1)
#print(extracted_data)

# Split the dataset into features (essays) and target variable (scores)
X = extracted_data.drop(['essay_id', 'Score', 'Essay', 'cleaned_essay', 'tokenized_essay'], axis=1)
y = extracted_data['Score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

# Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Evaluate the model
print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error (RMSE):', rmse)


# Read the input essay from a text file
file_path = '/content/drive/MyDrive/project/essay/essay1.txt'  # Replace with the path to your input essay text file
with open(file_path, 'r') as file:
    essay = file.read()

# Predict the score for the input essay
new_essay_features = vectorizer.transform([clean_text(essay)])
predicted_score = model.predict(new_essay_features)[0]
rounded_predicted_score = round(predicted_score, 1)

print(f"Score: {rounded_predicted_score}")
