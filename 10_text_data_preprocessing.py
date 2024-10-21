import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample movie reviews dataset
reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "The acting was terrible and the plot made no sense.",
    "Great special effects, but the story was lacking.",
    "I fell asleep halfway through. Very boring.",
    "A masterpiece of modern cinema. Highly recommended!"
]

# Create a DataFrame
df = pd.DataFrame({'review': reviews})

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Apply preprocessing to the reviews
df['processed_tokens'] = df['review'].apply(preprocess_text)

# Print original and processed reviews
print("Original vs Processed Reviews:")
for i, row in df.iterrows():
    print(f"\nOriginal: {row['review']}")
    print(f"Processed: {row['processed_tokens']}")

# Function to plot word frequency
def plot_word_frequency(tokens_list, top_n=10):
    all_tokens = [token for tokens in tokens_list for token in tokens]
    word_freq = Counter(all_tokens)
    top_words = dict(word_freq.most_common(top_n))
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot word frequency
plot_word_frequency(df['processed_tokens'])

# Function to convert tokens back to text
def tokens_to_text(tokens):
    return ' '.join(tokens)

# Convert processed tokens back to text
df['processed_text'] = df['processed_tokens'].apply(tokens_to_text)

# Print processed text
print("\nProcessed Text:")
for text in df['processed_text']:
    print(text)

# Function to get sentiment scores (simple approach)
def get_sentiment_score(tokens):
    positive_words = set(['good', 'great', 'excellent', 'fantastic', 'love', 'enjoy', 'recommend'])
    negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'boring'])
    
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    
    return positive_count - negative_count

# Apply sentiment scoring
df['sentiment_score'] = df['processed_tokens'].apply(get_sentiment_score)

# Print sentiment scores
print("\nSentiment Scores:")
for i, row in df.iterrows():
    print(f"Review: {row['review']}")
    print(f"Sentiment Score: {row['sentiment_score']}")
    print()

# Visualize sentiment scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(df)), df['sentiment_score'])
plt.title('Sentiment Scores for Movie Reviews')
plt.xlabel('Review Index')
plt.ylabel('Sentiment Score')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()