import classification_algorithms
from data_cleaning import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
import argparse


def report_metrics(classifier_name, y_true, y_pred):
    accuracy = metrics.classification_report(y_true, y_pred, 
                                        target_names=preprocess.CATEGORY_LABELS, 
                                        zero_division=0,
                                        output_dict=True)['accuracy']
    print(f'{classifier_name}: {round(accuracy * 100, 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeling', type=int, help='specify modeling approach to use options: tfidf(0), lda(1), w2v(3), bert(4)')
    parser.add_argument('--classifier', type=int, help='specify the type of classification to use. options: nb(0), dt(1), knn(2)')

    modeling_approaches = [preprocess.tf_idf_features, preprocess.topic_modeling_lda,
                           preprocess.w2v_word_embeddings, preprocess.bert_output_logits]
    classifiers = [classification_algorithms.naive_bayes_classifier,
                   classification_algorithms.decision_tree_classifier,
                   classification_algorithms.knn_classifier]

    args = parser.parse_args()

    modeling_fun = modeling_approaches[args.modeling]
    classification = classifiers[args.classifier]

    training_set, labels = preprocess.get_training_date(feature_modeling_fun=modeling_fun)
    # print('size', training_set.shape)
    X_train, X_test, y_train, y_test = train_test_split(training_set, labels,
                                                        test_size=.3,
                                                        random_state=42)

    if args.classifier == 0 and args.modeling in [2, 3]:
        classifier, name = classification(X_train, y_train, use_pipeline=True)
    else:
        classifier, name = classification(X_train, y_train)


    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    print(f'Modeling Approach: {str(modeling_fun)}')
    report_metrics(f'{name} Classifier Training Accuracy', y_train, y_train_pred)
    report_metrics(f'{name} Classifier Testing Accuracy', y_test, y_test_pred)



