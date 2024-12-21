import pandas as pd
from sklearn.model_selection import train_test_split

X_train = None
X_test = None
y_train = None
y_test = None
features = None

def process_data():
    train = pd.read_csv('data/train.csv').drop(columns=['Name', 'Ticket', 'Cabin']).dropna(
        subset=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    )
    # test data really only important for submitting to kaggle
    #test = pd.read_csv('data/test.csv').drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket']).dropna(
    #    subset=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    #)

    # Divide Age into categorical columns
    train['young_age'] = (train['Age'] <= 30).astype(int)
    train['medium_age'] = ((train['Age']) > 30 & (train['Age'] <= 60)).astype(int)
    train['old_age'] = (train['Age'] > 60).astype(int)
    train = train.drop(columns=['Age'])

    # Divide Fare into categorical columns
    train['average_fare'] = (train['Fare'] <= 52).astype(int)
    train['larger_fare'] = (train['Fare'] > 52).astype(int)
    train = train.drop(columns=['Fare'])
    X = train.to_numpy()
    y = train['Survived'].to_numpy()

    global X_train, X_test, y_train, y_test, features
    features = train.columns.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69, test_size=0.25, shuffle=True)