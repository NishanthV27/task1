import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("D:\intern\Titanic-Dataset.csv")


print(df.head())
print(df.info())
print(df.isnull().sum())   



df['Age'] = df['Age'].fillna(df['Age'].median())


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df = df.drop(['Cabin'], axis=1)



df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.show()

df = df[df['Fare'] < 300]
print(df.head(), "\n")
print(df.info(), "\n")