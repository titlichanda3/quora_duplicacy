import streamlit as st
import joblib
from gensim.models import Word2Vec
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
import nltk

# --- Paste the complete create_features_for_prediction function from above here ---


# It's good practice to download this once and define STOP_WORDS globally
# so it's not recreated every time the function is called.
nltk.download('stopwords')
STOP_WORDS = stopwords.words("english")

def create_features_for_prediction(q1, q2, w2v_model):
    """
    Takes two raw question strings and a trained Word2Vec model,
    and returns a NumPy array of all 6022 engineered features.
    """

    # --- Internal Helper Function: Preprocessing ---
    def preprocess(q):
        q = str(q).lower().strip()
        # ... (all your replacement and cleaning logic remains the same)
        q = q.replace('%', ' percent')
        q = q.replace('$', ' dollar ')
        q = q.replace('₹', ' rupee ')
        q = q.replace('€', ' euro ')
        q = q.replace('@', ' at ')
        q = q.replace('[math]', '')
        q = q.replace(',000,000,000 ', 'b ')
        q = q.replace(',000,000 ', 'm ')
        q = q.replace(',000 ', 'k ')
        q = re.sub(r'([0-9]+)000000000', r'\1b', q)
        q = re.sub(r'([0-9]+)000000', r'\1m', q)
        q = re.sub(r'([0-9]+)000', r'\1k', q)
        contractions = {"ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
        q_decontracted = []
        for word in q.split():
            if word in contractions:
                word = contractions[word]
            q_decontracted.append(word)
        q = ' '.join(q_decontracted)
        q = q.replace("'ve", " have")
        q = q.replace("n't", " not")
        q = q.replace("'re", " are")
        q = q.replace("'ll", " will")
        q = BeautifulSoup(q, "lxml").get_text()
        pattern = re.compile('\W')
        q = re.sub(pattern, ' ', q).strip()
        return q

    # 1. Preprocess questions
    q1_processed = preprocess(q1)
    q2_processed = preprocess(q2)
    
    # 2. Calculate manual features
    q1_len = len(q1_processed)
    q2_len = len(q2_processed)
    q1_num_words = len(q1_processed.split(" "))
    q2_num_words = len(q2_processed.split(" "))
    w1 = set(map(lambda word: word.lower().strip(), q1_processed.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2_processed.split(" ")))
    word_common = len(w1 & w2)
    word_total = (len(w1) + len(w2))
    word_share = round(word_common / word_total, 2) if word_total != 0 else 0
    token_features = [0.0]*8
    q1_tokens = q1_processed.split()
    q2_tokens = q2_processed.split()
    if len(q1_tokens) > 0 and len(q2_tokens) > 0:
        SAFE_DIV = 0.0001
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    length_features = [0.0]*3
    if len(q1_tokens) > 0 and len(q2_tokens) > 0:
        length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
        length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
        strs = list(distance.lcsubstrings(q1_processed, q2_processed))
        if len(strs) != 0:
            length_features[2] = len(strs[0]) / (min(len(q1_processed), len(q2_processed)) + 1)
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1_processed, q2_processed)
    fuzzy_features[1] = fuzz.partial_ratio(q1_processed, q2_processed)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1_processed, q2_processed)
    fuzzy_features[3] = fuzz.token_set_ratio(q1_processed, q2_processed)

    all_manual_features = [q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_total, word_share] + \
                           token_features + length_features + fuzzy_features

    # 3. Create Word2Vec features
    def vectorize_sentence(sentence, model):
        words = str(sentence).lower().split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if not word_vectors:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    q1_vec = vectorize_sentence(q1_processed, w2v_model)
    q2_vec = vectorize_sentence(q2_processed, w2v_model)

    # 4. Combine all features into a single array
    final_features = np.concatenate((all_manual_features, q1_vec, q2_vec))
    
    return final_features.reshape(1, -1)

# Load both of your pre-trained models
rf_model = joblib.load('rf_model.pkl')
w2v_model = Word2Vec.load("word2vec.model")

st.title('Quora Duplicate Question Detector')

question1 = st.text_input("Enter the first question:")
question2 = st.text_input("Enter the second question:")

if st.button('Predict'):
    if question1 and question2:
        # Create the full feature set using both questions and the w2v_model
        query_point = create_features_for_prediction(question1, question2, w2v_model)

        # Use the Random Forest model to predict
        prediction = rf_model.predict(query_point)[0]

        # Display the result
        if prediction == 1:
            st.success("These questions are likely duplicates!")
        else:
            st.error("These questions seem to be different.")
    else:
        st.warning("Please enter both questions.")
