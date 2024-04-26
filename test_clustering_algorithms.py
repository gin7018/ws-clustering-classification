import argparse
import clustering_algorithms
from data_cleaning import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeling', type=int, help='specify modeling approach to use. options: tfidf(0), lda(1), w2v(3), bert(4)')
    parser.add_argument('--algo', type=int, help='specify the type of clustering algorithm to use. options: kmeans(0), dbscan(1)')

    modeling_approaches = [preprocess.tf_idf_features, preprocess.topic_modeling_lda,
                           preprocess.w2v_word_embeddings, preprocess.bert_output_logits]
    clustering_algo = [clustering_algorithms.kmeans,
                       clustering_algorithms.dbscan]
    args = parser.parse_args()

    modeling_fun = modeling_approaches[args.modeling]

    training_set, labels = preprocess.get_training_date(feature_modeling_fun=modeling_fun)
    X_train, X_test, y_train, y_test = train_test_split(training_set, labels,
                                                        test_size=.3,
                                                        random_state=42)
    if args.algo == 0:
        for count in range(8, 30):
            model, name = clustering_algo[args.algo](X_train, y_train, n_clusters=count)
            training_labels = model.fit_predict(X_train)
        
            silhouette_coeff = metrics.silhouette_score(X_train, training_labels, metric='euclidean')
            print(f"clusters={count}; sc={silhouette_coeff}")
    else:
        counts = [0.5, 1.0, 1.1, 1.2, 1.3]
        for count in counts:
            model, name = clustering_algo[args.algo](X_train, max_distance=count)
            training_labels = model.labels_
            if len(set(training_labels)) == 1:
                continue
            silhouette_coeff = metrics.silhouette_score(X_train, training_labels, metric='euclidean')
            print(f"esp={count}; sc={silhouette_coeff}")

    
        