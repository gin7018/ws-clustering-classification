import classifiers
import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics


def report_metrics(classifier_name, y_true, y_pred):
    accuracy = metrics.classification_report(y_true, y_pred, 
                                        target_names=preprocess.CATEGORY_LABELS, 
                                        zero_division=0,
                                        output_dict=True)['accuracy']
    print(f'{classifier_name}: {round(accuracy * 100, 2)}')


if __name__ == '__main__':
    modeling_fun = preprocess.bert_output_logits

    cleaned_description, labels = preprocess.get_training_date(feature_modeling_fun=modeling_fun)
    X_train, X_test, y_train, y_test = train_test_split(cleaned_description, labels,
                                                        test_size=.3,
                                                        # train_size=.10,
                                                        random_state=42)

    classifier, name = classifiers.naive_bayes_classifier(X_train, y_train)

    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    report_metrics(f'{name} Classifier Training Accuracy', y_train, y_train_pred)
    report_metrics(f'{name} Classifier Testing Accuracy', y_test, y_test_pred)



