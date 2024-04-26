from sklearn.cluster import KMeans, DBSCAN

def kmeans(X, y, n_clusters=8):
    kmeans_algorithm = KMeans(random_state=42, 
                              n_clusters=n_clusters,
                              n_init=10).fit(X, y)
    return kmeans_algorithm, 'K-Means'


def dbscan(X, max_distance=1):
    dbscan_algorithm = DBSCAN(eps=max_distance).fit(X)
    return dbscan_algorithm, 'DB-Scan'