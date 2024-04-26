import clustering_algorithms
from data_cleaning import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    modeling_fun = preprocess.tf_idf_features

    training_set, labels = preprocess.get_training_date(feature_modeling_fun=modeling_fun)
    print('size', training_set.shape)
    X_train, X_test, y_train, y_test = train_test_split(training_set, labels,
                                                        test_size=.3,
                                                        random_state=42)
    
    model = 'km'
    if model == 'km':
        km_label = clustering_algorithms.kmeans(X_train).fit_predict(X_test)
        km_sc_score = metrics.silhouette_score(X_test, km_label, metric='euclidean')
        print(f"The silhouette coefficient score of KMeans is {km_sc_score}")
        