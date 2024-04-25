import pandas as pd
from pymongo import MongoClient

# Connect to the MongoDB server running on localhost
client = MongoClient('localhost', 27017)

# Access a specific database
db = client['your_database_name']

def extract():
    df = pd.read_csv('pa04.apirecord.csv')
