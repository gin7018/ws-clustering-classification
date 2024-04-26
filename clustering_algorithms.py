from sklearn.cluster import KMeans, DBSCAN

def kmeans(X):
    kmeans_algorithm = KMeans(random_state=42).fit(X)
    return kmeans_algorithm


def dbscan(X):
    dbscan_algorithm = DBSCAN().fit(X)
    return dbscan_algorithm