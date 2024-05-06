import pymongo
import logging
import pymongo.errors
import threading

class Bench:
    mongodb: str

    def __init__(self):
        self.mongodb = "mongodb://localhost:27017/"
        self.init_table()

    def init_table(self):
        try:
            client = self.get_db_conn()
            db = client['genesis']
            user = db['user']
            user.create_index("user_id")
            score = db['score']
            score.create_index([("timestamp", pymongo.DESCENDING)])

        except pymongo.errors.PyMongoError as e:
            logging.error(e)

    def get_db_conn(self) -> pymongo.MongoClient:
        client = pymongo.MongoClient(self.mongodb)
        return client
    
database_instance: Bench = None
init_completed_event = threading.Event()

def init_database():
    global database_instance
    database_instance = Bench()

def get_db_conn():
    global database_instance
    return database_instance.get_db_conn()