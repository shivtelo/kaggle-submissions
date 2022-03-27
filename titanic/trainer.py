import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
file_path = os.path.dirname(__file__)
basepath = file_path + '/resources/'

train = basepath + 'train.csv'
test = basepath + 'test.csv'
output = basepath + 'test_result.csv'

train_df = pd.read_csv(train)

feature_cols = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age']

outputColumns = ['Survived']

train_df['Sex'].replace('male', 0, inplace=True)
train_df['Sex'].replace('female', 1, inplace=True)
train_df['Age'].fillna(0, inplace=True)

X = train_df.loc[:, feature_cols]
Y = train_df.loc[:, outputColumns]

logreg = LogisticRegression()

logreg.fit(X, Y)
test_df = pd.read_csv(test)
test_df['Sex'].replace('male', 0, inplace=True)
test_df['Sex'].replace('female', 1, inplace=True)

filtered_df = test_df[test_df.Age.isnull()]
test_df = test_df[test_df.Age.notnull()]
X_new = test_df.loc[:, feature_cols]
new_predict_class = logreg.predict(X_new)
kaggle_data = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': new_predict_class}).set_index('PassengerId')
kaggle_data2 = pd.DataFrame({'PassengerId': filtered_df.PassengerId, 'Survived': 0}).set_index('PassengerId')
kaggle_data = pd.concat([kaggle_data, kaggle_data2])
kaggle_data.to_csv(output)
