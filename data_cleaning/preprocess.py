import re
import time

import gensim.corpora as corpora
import nltk
import numpy as np
from gensim.models import LdaModel
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast, DistilBertModel

CATEGORY_LABELS = ['Enterprise',  'Government', 'Photos', 'Travel', 'Education', 'Shopping', 'Financial',
                   'Payment',  'Video', 'Email', 'Telephony', 'Social', 'Music',
                   'Mapping', 'Reference', 'Advertising', 'Science', 'Messaging', 'Internet']


def clean_data(descriptions):
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    wn = WordNetLemmatizer()

    cleaned = []
    for text in descriptions:
        text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        text = [wn.lemmatize(word) for word in text if word not in stopwords]
        text = ' '.join(text)
        cleaned.append(text)
    return cleaned


def standardize_labels(categories):
    return [CATEGORY_LABELS.index(category) for category in categories]


def tf_idf_features(data):
    tf_idf = TfidfVectorizer()
    tf_idf.fit_transform(data)
    data_tf = tf_idf.transform(data)
    print(data_tf.shape)
    return data_tf


def topic_modeling_lda(data):
    tokens = [text.split() for text in data]
    word_id = corpora.Dictionary(tokens)
    corpus = [word_id.doc2bow(text) for text in tokens]
    topic_count = len(CATEGORY_LABELS)

    lda_model = LdaModel(corpus=corpus,
                         id2word=word_id,
                         num_topics=topic_count,
                         random_state=42,
                         passes=12,
                         alpha='auto',
                         per_word_topics=True)

    # print(lda_model.print_topics())
    # lda model coherence score = 0.45344603559759217
    print('lda model training done')
    feature_vectors = []
    for text_bow in corpus:
        topics = lda_model.get_document_topics(text_bow, minimum_probability=0.0)
        feature_vector = [topic[1] for topic in topics]
        feature_vectors.append(feature_vector)

    return feature_vectors


def w2v_word_embeddings(data):
    SEQUENCE_LEN = 50
    EMBEDDING_SIZE = 100

    embeddings_dict = dict()
    with open(f'data_cleaning/embedding-models/glove.6B.{EMBEDDING_SIZE}d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embeddings_dict[word] = vector
    print('word embeddings loaded')

    tokens = [text.split()[:SEQUENCE_LEN] for text in data]
    embeddings = []
    for text in tokens:
        vector = [embeddings_dict[word] for word in text if word in embeddings_dict]
        if len(vector) != SEQUENCE_LEN: # padding
            vector += [[0.0] * EMBEDDING_SIZE for _ in range(SEQUENCE_LEN-len(vector))]
        embeddings.append(vector)
    embeddings = np.array(embeddings).reshape((len(data), SEQUENCE_LEN*EMBEDDING_SIZE))
    return embeddings


def bert_output_logits(data):
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',TOKENIZERS_PARALLELISM=False)
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    print('bert model loaded')

    def get_output(batch):
        tokens = bert_tokenizer(batch, return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=80,
                                add_special_tokens=True)
        outputs = bert_model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    start = time.time()
    x_train = []
    k, i, j = 1, 0, 380
    while j <= 3800:
        output = get_output(data[i:j])
        x_train.append(output)
        del output
        print(f'batch {k} done.')
        i = j
        j += 380
        k += 1
    
    x_train = np.array(x_train)
    x_train = x_train.reshape(-1, x_train.shape[-1])
    print(f'bert outputs obtained in {time.time() - start} seconds')

    return x_train


def get_training_date(filename='data_cleaning/training_data.npy', feature_modeling_fun=tf_idf_features):
    x_train = np.load(filename, allow_pickle=True)
    np.random.shuffle(x_train)

    print('loaded')

    ws_descriptions = [str(text) for text in x_train[:, 1] if text is not np.nan]
    ws_categories = x_train[:, 2]

    cleaned_descriptions = clean_data(ws_descriptions)
    transformed_descriptions = feature_modeling_fun(cleaned_descriptions)
    labels = standardize_labels(ws_categories)
    return transformed_descriptions, labels


if __name__ == '__main__':
    x_train = np.load('training_data.npy', allow_pickle=True)
    print('training data loaded')

    ws_descriptions = [str(text) for text in x_train[:, 1] if text is not np.nan]
    ws_categories = x_train[:, 2]

    cleaned_descriptions = clean_data(ws_descriptions)
    start = time.time()
    x_train = bert_output_logits(cleaned_descriptions)
    print('shape of x_train: ', x_train.shape)
    print(f'outputs obtained in {time.time() - start} seconds')
