from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler


def naive_bayes_classifier(X, y, use_pipeline=False):

    classifier = MultinomialNB()
    if use_pipeline:
        classifier = Pipeline([('normalizing', MinMaxScaler()), ('naive-bayes', classifier)])

    classifier.fit(X, y)
    return classifier, 'Naive Bayes'


def decision_tree_classifier(X, y):
    classifier = DecisionTreeClassifier(min_samples_split=10,
                                        # max_depth=10,
                                        criterion='entropy',
                                        random_state=42)
    classifier.fit(X, y)
    print('max depth = ', classifier.get_depth())
    return classifier, 'Decision Tree'


def knn_classifier(X, y):
    classifier = KNeighborsClassifier()
    classifier.fit(X, y)
    return classifier, 'K Nearest-Neighbors'

