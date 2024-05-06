from be.model import bench

class DBConn():

    def __init__(self) -> None:
        self.client = bench.get_db_conn()
        self.db = self.client["genesis"]


    def user_id_exist(self, user_id):
        user_col = self.db['user']
        doc = user_col.find_one({'user_id': user_id})
        if doc is None:
            return False
        else:
            return True