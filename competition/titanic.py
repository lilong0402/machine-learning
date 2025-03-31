import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('./titanic'):
    print(dirname)
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 加载数据
train_data = pd.read_csv("./titanic/train.csv")
# print(train_data.head())
test_data = pd.read_csv("./titanic/test.csv")
print(test_data.head())


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
print(f"y:{y}")
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
print(f"X:{X}")
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


