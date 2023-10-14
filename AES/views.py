from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.views import View
from io import StringIO
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet


def index(request):
    print(request)
    return render(request,"main.html")

class score_essay(View):
    def get(self,request):
        print(request)
        return render(request,"main.html")
    def post(self,request):
        print(request.POST)
        print(request.POST.get('essay'))
    
       
        if request.POST.get('user_selection') == "Inputstr":
           essay_texts=pd.read_csv( StringIO(request.POST.get('essay')), sep='\t', header=None)
        else:
           handle_uploaded_file(request.FILES['pdfFile'])
           essay_texts = pd.read_csv("C:\\Users\\rosemol\\Desktop\\AES_PROJECT\\grade\\name.txt", sep='\t', header=None)
           
        #  Cleaning and preprocessing
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        cleaned_essays = []
        for essay in essay_texts[0].values:
            # Remove punctuation and convert to lowercase
            essay = essay.lower()
            essay = "".join([c for c in essay if c.isalpha() or c.isspace()])

            # Tokenization
            tokens = word_tokenize(essay)

            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            # Reconstruct the essay
            cleaned_essay = " ".join(tokens)
            cleaned_essays.append(cleaned_essay)

        # Feature extraction
        df = pd.DataFrame({'essay': cleaned_essays})
    
        # Word count
        df['word_count'] = df['essay'].apply(lambda x: len(x.split()))

        # Sentence count
        df['sentence_count'] = df['essay'].apply(lambda x: len(nltk.sent_tokenize(x)))

        # Average word length
        df['avg_word_length'] = df['essay'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))

        # Coherence (Word Embeddings)

        # Load word embeddings (e.g., GloVe)
        word_embeddings_path = "glove.6B.50d.txt"
        # Load word embeddings into a dictionary
        word_embeddings = {}
        with open(word_embeddings_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    embedding = np.array(values[1:], dtype='float32')
                    word_embeddings[word] = embedding
                except ValueError:
                    continue

        # Tokenize essays
        essays_tokens = [word_tokenize(essay) for essay in df['essay']]

        # Compute coherence scores using cosine similarity between word embeddings
        coherence_scores = []
        for essay_tokens in essays_tokens:
            essay_embeddings = [word_embeddings.get(token) for token in essay_tokens if token in word_embeddings]
            if len(essay_embeddings) > 1:
                essay_similarity_matrix = cosine_similarity(essay_embeddings)
                coherence_score = np.mean(essay_similarity_matrix)
            else:
                coherence_score = 0.0
            coherence_scores.append(coherence_score)

        df['coherence_score'] = coherence_scores
        # Lexical features means vocabulary richness
        df['unique_word_count'] = df['essay'].apply(lambda x: len(set(x.split())))
        df['lexical_diversity'] = df['unique_word_count'] / df['word_count']

        # Compute semantic similarity (Cosine similarity) between essays
        semantic_similarity_scores = []
        for i in range(len(essays_tokens)):
            for j in range(i + 1, len(essays_tokens)):
                essay1_tokens = essays_tokens[i]
                essay2_tokens = essays_tokens[j]

                essay1_embeddings = [word_embeddings.get(token) for token in essay1_tokens if token in word_embeddings]
                essay2_embeddings = [word_embeddings.get(token) for token in essay2_tokens if token in word_embeddings]

                if len(essay1_embeddings) > 0 and len(essay2_embeddings) > 0:
                    similarity_matrix = cosine_similarity(essay1_embeddings, essay2_embeddings)
                    semantic_similarity_score = np.mean(similarity_matrix)
                else:
                    semantic_similarity_score = 0.0

                semantic_similarity_scores.append(semantic_similarity_score)

        # Create an empty array to hold the scores for all essay pairs
        pairwise_scores = np.zeros((len(df), len(df)))

        # Fill the pairwise_scores array with the semantic similarity scores
        pairwise_index = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                pairwise_scores[i, j] = semantic_similarity_scores[pairwise_index]
                pairwise_scores[j, i] = semantic_similarity_scores[pairwise_index]
                pairwise_index += 1

        # Add semantic similarity scores to the DataFrame
        df['semantic_similarity_score'] = pairwise_scores.mean(axis=1)
        # TF-IDF features
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_features = tfidf_vectorizer.fit_transform(df['essay'])

        # Convert sparse matrix to DataFrame
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Concatenate the TF-IDF features with the original DataFrame
        df = pd.concat([df, tfidf_df], axis=1)

        # Define weights for each feature
        word_count_weight = 0.2
        sentence_count_weight = 0.1
        avg_word_length_weight = 0.15
        coherence_score_weight = 0.2
        lexical_diversity_weight = 0.15
        semantic_similarity_score_weight = 0.1
        tfidf_weights = np.ones(tfidf_df.shape[1]) * 0.1  # Equal weights for all TF-IDF features

        # Calculate scores for each essay
        df['score'] = (
            df['word_count'] * word_count_weight +
            df['sentence_count'] * sentence_count_weight +
            df['avg_word_length'] * avg_word_length_weight +
            df['coherence_score'] * coherence_score_weight +
            df['lexical_diversity'] * lexical_diversity_weight +
            df['semantic_similarity_score'] * semantic_similarity_score_weight +
            np.dot(tfidf_df.values, tfidf_weights)
        )

        # Normalize scores to a 0-100 scale
        min_score = df['score'].min()
        max_score = df['score'].max()
        df['normalized_score'] = (df['score'] - min_score) / (max_score - min_score) * 100

        # Convert normalized score to the range 0-10
        df['normalized_score_0_to_10'] = (df['normalized_score'] / 100) * 10

        # Sort essays by the normalized score in descending order
        df = df.sort_values('normalized_score_0_to_10', ascending=False)

        # Calculate the overall score
        overall_score = df['normalized_score_0_to_10'].mean()

        # Round off the overall score
        rounded_score = round(overall_score,1)

        # Print the rounded overall score
        print(" Score:", rounded_score)
           
           
        response=HttpResponse(rounded_score)
        return response 
   
def handle_uploaded_file(f):
    with open("name.txt", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)