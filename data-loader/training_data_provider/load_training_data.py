import numpy as np
import pandas as pd
from pymongo import MongoClient

client = MongoClient('localhost', 27017)

db = client['pa04']
collection = db.get_collection('apirecords')


def load():
    category_df = pd.read_csv('pa04.apirecords.csv')
    training_data = []

    for idx, row in category_df.iterrows():
        category = row['_id']
        results = collection.find({
            'category': category
        }, {
            'category': 1,
            'description': 1,
            'name': 1,
            '_id': 0
        })
        for el in results:
            training_data.append([
                el['name'],
                el['description'],
                el['category']
            ])
    print(f'retrieved {len(training_data)} documents')
    training_data = np.array(training_data)
    np.save('training_data', training_data)
    print('saved')


if __name__ == '__main__':
    load()
