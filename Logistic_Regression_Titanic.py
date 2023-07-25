import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing

# Read the following link and complete this homework. https://www.codemag.com/Article/1711091/Implementing-Machine-Learning-Using-Python-and-Scikit-learn

# Make sure to install scikit-learn and Pandas


url = "https://gist.githubusercontent.com/mkzia/aa4f293661dba857b8c4459c0095ac95/raw/8075037f6f7689a1786405c1bc8ea9471d3aa9c3/train.csv"


def step1():
    """
    # Step 1: Getting the Titanic Dataset
    Return a dataframe containing the Titantic dataset from the following URL
    # URL: https://gist.githubusercontent.com/mkzia/aa4f293661dba857b8c4459c0095ac95/raw/8075037f6f7689a1786405c1bc8ea9471d3aa9c3/train.csv

    """
    # BEGIN SOLUTION
    df = pd.read_csv(url)
    # END SOLUTION
    return df


def step2(df):
    """
    # Step 2: Clean data
    Modify df to drop the following columns:
    PassengerId
    Name
    Ticket
    Cabin
    Hint: Just pass all the columns to the .drop() method as an array
    return dataframe
    """
    # BEGIN SOLUTION
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    # END SOLUTION
    return df


def step3(df):
    """
    # Step 3: Drop NaNs and reindex
    You want to reindex so your index does not have missing values after you drop the NaNs. Remember, index is used
    to access a row. Notice how many rows you dropped!
    Modify df to drop NaNs and reindex
    return dataframe
    """
    # BEGIN SOLUTION
    df = df.dropna().reset_index(drop=True)
    # END SOLUTION
    return df


def step4(df):
    """
    # Step 4: Encoding the Non-Numeric Fields
    Encode text fields to numbers
    Modify df to encode Sex and Embarked to encoded values.
    return dataframe
    """
    # BEGIN SOLUTION
    df = df.replace({"male": 1, "female": 0, "S": 0, "C": 1, "Q": 2})
    # END SOLUTION
    return df


def step5(df):
    """
    # Step 5: Making Fields Categorical
    Turn values that are not continues values into categorical values
    Modify df to make Pclass, Sex, Embarked, and Survived a categorical field
    return dataframe
    """
    # BEGIN SOLUTION
    category_col = ["Pclass", "Sex", "Embarked", "Survived"]

    for col in df.columns:
        if col in category_col:
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(df[col].dtype)

    # END SOLUTION
    return df


def step6(df):
    """
    1. Split dataframe into feature and label
    2. Do train and test split; USE: random_state = 1
    3. Return accuracy and confusion matrix
    4. Use LogisticRegression() for classification

    Use metrics.confusion_matrix to calculate the confusion matrix
    # https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # IMPORTANT !!!!
    # https://stackoverflow.com/questions/56078203/why-scikit-learn-confusion-matrix-is-reversed

    From the confusion matrix get TN, FP, FN, TP

    return --> accuracy, TN, FP, FN, TP;
    Hint: round accuracy to 4 decimal places

    """
    # BEGIN SOLUTION
    random_state = 1
    test_size = 0.25

    # df.head()

    # df.columns
    feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    label_columns = "Survived"

    # X = df.drop([label_columns], axis=1)
    # X = df(feature_columns, axis=1)
    X = df.loc[:, feature_columns]
    y = df[label_columns]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df[label_columns]
    )
    # scaler = preprocessing.StandardScaler().fit(train_X)
    # train_X = scaler.transform(train_X)
    # test_X = scaler.transform(test_X)

    # model
    lr = LogisticRegression(random_state=random_state)
    # model training
    lr.fit(train_X, train_y)

    # model testing
    pred_y = lr.predict(test_X)

    # accuracy
    accuracy = accuracy_score(test_y, pred_y)
    accuracy = round(accuracy, 4)

    # confusion matrix
    cm = confusion_matrix(test_y, pred_y)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # END SOLUTION
    return accuracy, TN, FP, FN, TP
