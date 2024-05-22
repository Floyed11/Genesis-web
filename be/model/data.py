import os
import subprocess
import pandas as pd
import pandas as pd
import pickle
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn, optim

def train_new_model(iterations):
    
    print("new training start!")
    iterations = int(iterations)
    os.chdir('./model/genesis')
    #subprocess.run(['scrapy', 'crawl', 'spec'])
    os.chdir('../..')

    print("spider success!")
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

    csv_file_path = './model/genesis/spider_spec_cfp2017.csv'
    data = pd.read_csv(csv_file_path)

    # data = data.assign(Cache_L1_I = lambda x:x["Cache L1"].str.split(' ').str[0] + ' ' + x["Cache L1"].str.split(' ').str[1],
    #                    Cache_L1_D = lambda x:x["Cache L1"].str.split(' ').str[4] + ' ' + x["Cache L1"].str.split(' ').str[5])
    data = data.assign(Cache_L1_I = trans_L1_I, Cache_L1_D = trans_L1_D)
    del data["Cache L1"]

    # data = data.assign(Cache_L2_ID = lambda x:x["L2"].str.split(' ').str[0] + ' ' + x["L2"].str.split(' ').str[1])
    data = data.assign(Cache_L2_ID = trans_L2)
    del data["L2"]

    # data = data.assign(Cache_L3_ID = lambda x:x["L3"].str.split(' ').str[0] + ' ' + x["L3"].str.split(' ').str[1])
    data = data.assign(Cache_L3_ID = trans_L3)
    data.loc[:, "L3_shared"] = data.apply(L3_shared_temp, axis=1)
    del data["L3"]

    data = data.assign(Memory_num = trans_memory)
    del data["Memory"]

    data = data.assign(Storage_num = trans_storage)
    del data["Storage"]

    data.to_csv('./model/clean.csv', index=False)


    columns = data.columns

    columns_num_flag = {columns[i]: False for i in range(len(columns))}
    columns_num_flag['Max MHz'] = True
    columns_num_flag['Nominal'] = True
    '''可能使用数值也可能使用布尔'''
    # columns_num_flag['Cache L1'] = True
    # columns_num_flag['L2'] = True
    # columns_num_flag['L3'] = True
    # columns_num_flag['Memory'] = True
    # columns_num_flag['Storage'] = True
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
    y = data.loc[:, 'Base Results']
    for column in X.columns:
        if columns_num_flag[column]:  
            X_num_table[column] = X[column]
            # print(column)
        else:   
            X_bool_table = pd.concat([X_bool_table, pd.get_dummies(X[column])], axis=1)

    # test
    with open('./model/columns_new.pkl', 'wb') as f:
        pickle.dump(X_bool_table.columns, f)


    X_bool_table = X_bool_table.astype(int)
    X_after = pd.concat([X_bool_table, X_num_table], axis=1)
    # X_after.to_csv('output_X.csv', index=False)

    # y.to_csv('output_y.csv', index=False)

    X_after.to_csv('output_X_test.csv', index=False)

    y.to_csv('output_y_test.csv', index=False)


    IN_FEATURES = 1740
    ITERATION = iterations

    '''输入数据集'''
    data_X = X_after

    data_X = data_X.apply(pd.to_numeric, errors='coerce')  # 将无法转换为数值的值设置为NaN
    data_X = data_X.fillna(0)  # 将NaN值设置为0

    IN_FEATURES = data_X.shape[1]
    print("IN_FEATURES: ", IN_FEATURES)

    data_y = y

    data_y = data_y.apply(pd.to_numeric, errors='coerce')  # 将无法转换为数值的值设置为NaN
    data_y = data_y.fillna(0)  # 将NaN值设置为0

    '''查看输出'''
    X = torch.FloatTensor(data_X.values)
    y = torch.FloatTensor(data_y.values).unsqueeze(1)

    X = (X - X.mean()) / X.std()



    # 定义模型
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, 64),  # 输入层
        nn.ReLU(),  # 激活函数
        nn.Linear(64, 1)  # 输出层
    )

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # 训练

    for epoch in range(ITERATION):
        if (epoch + 1) % 1000 == 0:  # 每1000次迭代打印一次
                print('Epoch: {:.2f}%'.format((epoch + 1)/ITERATION * 100))

        # 前向传播
        y_pred = model(X)

        # 计算损失
        loss = loss_fn(y_pred, y)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 清零梯度
        optimizer.zero_grad()

        
    # 保存模型
    torch.save(model.state_dict(), './model/model_parameters_new.pth')