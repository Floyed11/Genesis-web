
import torch
from torch import nn
import pandas as pd
import pymongo
from be.model import db_conn
import pickle
import os

class Pred(db_conn.DBConn):
    def __init__(self):
        IN_FEATURES = 1740
        try:
            with open('model/columns.pkl', 'rb') as f:
                print('load columns success')
                load_columns = pickle.load(f)
        except Exception as e:
            print(e)
        IN_FEATURES = len(load_columns) + 6
        self.model = nn.Sequential(
            nn.Linear(IN_FEATURES, 64),  # 输入层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 1)  # 输出层
        )
        try:
            self.path = 'model/model_parameters.pth'
            self.model.load_state_dict(torch.load(self.path))
            print('load model success')
        except Exception as e:
            print(e)
        db_conn.DBConn.__init__(self)

    def rename_key(self, data: dict, old_key: str, new_key: str):
        if old_key in data:
            data[new_key] = data.pop(old_key)
        else:
            print('rename key not found: ', old_key)

    def predict(self, data: dict):
        # find_result = self.db['trans_in'].find_one(sort=[('timestamp', pymongo.DESCENDING)], projection={'_id': 0, 'timestamp': 0})
        # print("find_result: ", find_result)
        # if find_result is None:
        #     return -1
        # print(data)
        self.rename_key(data, 'testSponsor', 'Test Sponsor')
        self.rename_key(data, 'cpuName', 'Cpu Name')
        self.rename_key(data, 'maxMHz', 'Max MHz')
        self.rename_key(data, 'nominal', 'Nominal')
        self.rename_key(data, 'oderable', 'Oderable')
        self.rename_key(data, 'cacheL1', 'Cache L1')
        self.rename_key(data, 'cacheL2', 'L2')
        self.rename_key(data, 'cacheL3', 'L3')
        self.rename_key(data, 'cacheOther', 'Cache Other')
        self.rename_key(data, 'memory', 'Memory')
        self.rename_key(data, 'storage', 'Storage')
        self.rename_key(data, 'hardwareOther', 'Hardware Other')
        self.rename_key(data, 'os', 'OS')
        self.rename_key(data, 'compiler', 'Compiler')
        self.rename_key(data, 'firmware', 'Firmware')
        self.rename_key(data, 'fileSystem', 'File System')
        self.rename_key(data, 'systemState', 'System State')
        self.rename_key(data, 'basePointers', 'Base Pointers')
        self.rename_key(data, 'peakPointers', 'Peak Pointers')
        self.rename_key(data, 'softwareOther', 'Software Other')
        self.rename_key(data, 'powerManagement', 'Power Management')
        self.rename_key(data, 'paralled', 'Paralled')
        self.rename_key(data, 'baseThreads', 'Base Threads')
        self.rename_key(data, 'enabledCores', 'Enabled Cores')
        self.rename_key(data, 'enabledChips', 'Enabled Chips')
        self.rename_key(data, 'threadsCore', 'Threads/Core')


        data_X = pd.DataFrame(data, index=[0])
        #print('data_X is:\n', data_X)
        data_X = self.clean_trans(data_X)
        data_X = data_X.apply(pd.to_numeric, errors='coerce')
        data_X = data_X.fillna(0)
        X = torch.FloatTensor(data_X.values)
        X = (X - X.mean()) / X.std()
        result_y = self.model(X)
        result_y = result_y.item()
        #print('result_y is:\n', result_y)
        result_y = round(result_y, 2)
        return result_y


    def update(self):
        pass

    def clean_trans(self, data: pd.DataFrame):

        # clean
        ERROR = -1  

        def L3_shared_temp(x):
            try:
                if(len(x["L3"].split(' ')) > 8):
                    return x["L3"].split(' ')[7] + ' ' + x["L3"].split(' ')[8] + ' ' + x["L3"].split(' ')[11]
                else:
                    return False
            except:
                return False
            
        def trans_L1_I(data):
            try:
                x = data["Cache L1"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try: 
                        num = row.split(' ')[0]
                        num = float(num)
                        r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                print(row)
                return ERROR

        def trans_L1_D(data):
            try:
                x = data["Cache L1"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try: 
                        num = row.split(' ')[4]
                        num = float(num)
                        r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                print(row)
                return ERROR

        def trans_L2(data):
            try:
                x = data["L2"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try:
                        num = row.split(' ')[0]
                        type = row.split(' ')[1]
                        if (type == 'MB'):
                            num = float(num) * 1024
                        else:
                            num = float(num)
                        r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                # print(row)
                return ERROR

        def trans_L3(data):
            try:
                x = data["L3"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try: 
                        num = row.split(' ')[0]
                        num = float(num)
                        r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                print(row)
                return ERROR

        def trans_memory(data):
            try:
                x = data["Memory"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try: 
                        num = row.split(' ')[0]
                        type = row.split(' ')[1]
                        if (type == 'TB'):
                            num = float(num) * 1024
                        else:
                            num = float(num)
                        r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                print(row)
                return ERROR
            
        def trans_storage(data):
            try:
                x = data["Storage"]
                r = pd.Series([])
                for row in x:
                    if row == 'redacted':
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                        continue
                    try: 
                        if 'x' in row:
                            num = row.split(' ')[2]
                            type = row.split(' ')[3]
                            if type == 'TB':
                                num = float(num) * 1024
                            else:
                                num = float(num)
                            r = pd.concat([r, pd.Series([num])], ignore_index=True)
                        else:
                            num = row.split(' ')[0]
                            type = row.split(' ')[1]
                            if type == 'TB':
                                num = float(num) * 1024
                            else:
                                num = float(num)
                            r = pd.concat([r, pd.Series([num])], ignore_index=True)
                    except Exception as e:
                        r = pd.concat([r, pd.Series([ERROR])], ignore_index=True)
                return r
            except Exception as e:
                print(e)
                print(row)
                return ERROR
            
        data = data.assign(Cache_L1_I = trans_L1_I, Cache_L1_D = trans_L1_D)
        del data["Cache L1"]

        data = data.assign(Cache_L2_ID = trans_L2)
        del data["L2"]

        data = data.assign(Cache_L3_ID = trans_L3)
        data.loc[:, "L3_shared"] = data.apply(L3_shared_temp, axis=1)
        del data["L3"]

        data = data.assign(Memory_num = trans_memory)
        del data["Memory"]

        data = data.assign(Storage_num = trans_storage)
        del data["Storage"]

        # print('after clean \n', data)
        # trans
        columns = data.columns

        columns_num_flag = {columns[i]: False for i in range(len(columns))}
        columns_num_flag['Max MHz'] = True
        columns_num_flag['Nominal'] = True
        columns_num_flag['Cache_L1_I'] = True
        columns_num_flag['Cache_L1_D'] = True
        columns_num_flag['Cache_L2_ID'] = True
        columns_num_flag['Cache_L3_ID'] = True
        columns_num_flag['Memory_num'] = True
        columns_num_flag['Storage_num'] = True

        columns_num_flag['Base Threads'] = True
        columns_num_flag['Enabled Cores'] = True
        columns_num_flag['Enabled Chips'] = True
        columns_num_flag['Threads/Core'] = True
        # Cache_L1_I,Cache_L1_D,Cache_L2_ID,Cache_L3_ID,L3_shared,Memory_num,Storage_num
        '''分离输入和输出'''
        X_num_table = pd.DataFrame()
        X_bool_table = pd.DataFrame()
        X = data.loc[:, 'Test Sponsor':'Threads/Core']
        try:
            with open('model/columns.pkl', 'rb') as f:
                load_columns = pickle.load(f)
            # print('load_columns\n', load_columns.duplicated())
            # for i in load_columns.duplicated():
            #     if i:
            #         print('duplicated')
        except Exception as e:
            print(e)
            return -3

        for column in X.columns:
            if columns_num_flag[column]:  
                X_num_table[column] = X[column]
            else:   
                try:
                # new_data_dummies = pd.get_dummies(X[column])
                # new_data_dummies = new_data_dummies.reindex(columns=load_columns[column], fill_value=0)
                # X_bool_table = pd.concat([X_bool_table, new_data_dummies], axis=1)
                    X_bool_table = pd.concat([X_bool_table, pd.get_dummies(X[column])], axis=1)
                except Exception as e:
                    print(e)
        #print('X_bool_table\n', X_bool_table)
        try:
            X_bool_table = X_bool_table.astype(int)
            #print('X_bool_table\n', X_bool_table)
            X_bool_table = X_bool_table.loc[:,~X_bool_table.columns.duplicated()]
            #print('X_bool_table\n', X_bool_table)
            X_bool_table = X_bool_table.reindex(columns=load_columns, fill_value=0)
        except Exception as e:
            print(e)
            return -4
        X_after = pd.concat([X_bool_table, X_num_table], axis=1)
        # print('X_after\n', X_after)

        return X_after