import pandas as pd


def dataprocess(pathtrain,pathtest):
    data = pd.read_csv(pathtrain).iloc[:, 1:]
    test = pd.read_csv(pathtest)
    ids = test.iloc[:, 0]
    test = test.iloc[:, 1:]
    # 删除 博客名称、标题这两列
    data.drop(columns=['Episode_Title'], inplace=True)
    test.drop(columns=['Episode_Title'], inplace=True)
    # 数值型
    data['Episode_Length_minutes'].fillna(data['Episode_Length_minutes'].median(), inplace=True)
    # ratio = data['Listening_Time_minutes'].mean() / data['Episode_Length_minutes'].mean()
    # data['Episode_Length_minutes'].fillna(data['Listening_Time_minutes'] / ratio,inplace=True)
    # data['Host_Popularity_percentage'].fillna(data['Host_Popularity_percentage'].median(), inplace=True)
    # data['Episode_Length_missing'] = data['Episode_Length_minutes'].isna().astype(float)
    data['Guest_Popularity_percentage'].fillna(data['Guest_Popularity_percentage'].median(), inplace=True)
    data['Number_of_Ads'].fillna(data['Number_of_Ads'].mode()[0], inplace=True)

    # 类别型
    # data['Genre'].fillna('Unknown', inplace=True)
    # data['Episode_Sentiment'].fillna(data['Episode_Sentiment'].mode()[0], inplace=True)
    # data['Publication_Day'].fillna(data['Publication_Day'].mode()[0], inplace=True)
    # data['Publication_Time'].fillna(data['Publication_Time'].mode()[0], inplace=True)
    # test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test['Listening_Time_minutes'] / ratio)
    test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)
    # test['Episode_Length_missing'] = test['Episode_Length_minutes'].isna().astype(float)
    test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)

    data = pd.get_dummies(data)
    test = pd.get_dummies(test).astype(float)
    target = "Listening_Time_minutes"
    train_labels = data[target]
    data_train = data.drop(columns=[target]).astype(float)
    # print(data_train.isnull().sum())
    return data_train,test,train_labels,ids
