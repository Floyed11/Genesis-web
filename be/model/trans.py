import pymongo
from pymongo.errors import PyMongoError
from be.model import db_conn
from be.model import error
from be.model import pred
import time

class Trans(db_conn.DBConn):

    def __init__(self):
        db_conn.DBConn.__init__(self)

    def trans_in(self, data: dict) -> (int, str, int):
        try:
            if data is None:
                return 200, "ok", -2
            else:
                data['timestamp'] = time.time()
                self.db['trans_in'].insert_one(data)

            del data['projectName']
            del data['projectObjective']
            del data['timestamp']
            del data['_id']
            p = pred.Pred()
            score = p.predict(data)
            if score < 0:
                return 513, "ok", -1
            else:
                self.db['score'].insert_one({'score': score, 'timestamp': time.time()})

        except PyMongoError as e:
            return 528, "{}".format(str(e)), -1
        except BaseException as e:
            return 530, "{}".format(str(e)), -1
        return 200, "ok", score
    
    def get_result(self) -> (int, str, int):
        try:
            find_result = self.db['score'].find_one(sort=[('timestamp', pymongo.DESCENDING)])
            if find_result is None:
                return 513, "ok", -1
            score = find_result['score']
            if score is None or score < 0:
                return 513, "ok", -1
            else:
                return 200, "ok", score
            
        except PyMongoError as e:
            return 528, "{}".format(str(e)), -1
        except BaseException as e:
            return 530, "{}".format(str(e)), -1
        return 200, "ok", score
