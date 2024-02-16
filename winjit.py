import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Specify the file path and encoding
file_path = r'F:\train1.csv'
file_encoding = 'latin1'  # or 'iso-8859-1' or any other suitable encoding

try:
    # Attempt to read the CSV file with specified encoding
    data = pd.read_csv(file_path, encoding=file_encoding)
    
    # Check if the required columns exist in the dataset
    if 'Sentiment' not in data.columns or 'News Headline' not in data.columns:
        raise ValueError("Required columns not found in the dataset.")
    
    # Tokenize text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['News Headline'])

    # Convert sentiment labels to numerical values
    y = data['Sentiment']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Save the classifier for future use
    dump(classifier, 'sentiment_analysis_classifier.joblib')

    # Take input from the user
    my_st = input("Enter your Statement: ")

    # Load the trained classifier
    classifier = load('sentiment_analysis_classifier.joblib')

    # Transform the input statement using the same vectorizer
    my_st_vectorized = vectorizer.transform([my_st])

    # Predict sentiment label for the input statement
    predicted_sentiment = classifier.predict(my_st_vectorized)

    # Print the predicted sentiment
    print(f'Predicted Sentiment: {predicted_sentiment[0]}')

except OSError as e:
    # Handle the case where reading the file failed
    print(f"Error reading the file: {e}")
except ValueError as e:
    # Handle the case where the required columns are not found
    print(f"Error: {e}")