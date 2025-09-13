import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
pd.options.display.max_rows = 9999
df = pd.read_csv('dataset/comments1.csv')

# # Check the first few rows
# print(df.head())

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to string in case of NaN
    text = str(text).lower()
    
    # Remove URLs and HTML
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove user handles and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Join back into a string
    return ' '.join(lemmatized_tokens)

# Apply cleaning to the comments column
df['cleaned_text'] = df['textOriginal'].apply(clean_text)

# Preview cleaned results
print("\n--- Data After Cleaning ---")
print(df[['textOriginal', 'cleaned_text']].head(10))

