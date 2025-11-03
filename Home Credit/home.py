import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

#Load Data
df = pd.read_csv('application_train.csv')
df.head()
  
#Mengecek tipe data
df.info()
  
#statistik
df.describe()
  
#cek missing value
df.isnull().sum()
  
#hapus kolom >50% kososng
for col in df.columns:
  if df[col].isnull().mean() > 0.5:
    df.drop(col, axis=1, inplace=True)
  
#mengisi missing value kolom numerik dengan median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

#mengisi missing value kolom kategorikal dengan nilai yang sering muncul
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

#cek apakah masih ada missing value
df.isnull().sum()
df.info()
df.describe()

#EDA(Exploratory Data Analysis)
#Statistik Dasa
df.describe()

#Cek frekuensi kategorikal: kolom occupation
df['OCCUPATION_TYPE'].value_counts()
df['NAME_CONTRACT_TYPE'].value_counts()

#Visualisasi sederhana
#histogram
import matplotlib.pyplot as plt
df['AMT_CREDIT'].hist(bins=50)
plt.show()

#bar chart
df['OCCUPATION_TYPE'].value_counts().plot(kind='bar')
plt.show()

df['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar')
plt.show()

#Encode kolom kategorikal
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

#Pisahkan fitur dan target
X = df.drop('TARGET', axis=1)
y = df['TARGET']

#Split data/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#melatih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evaluasi
y_pred = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:,1]

accuracy_score(y_val, y_pred), roc_auc_score(y_val, y_pred_prob), confusion_matrix(y_val, y_pred), classification_report(y_val, y_pred)

#ambil fitur penting
importances = model.feature_importances_
features = X.columns

#buat dataframe
feat_importance = pd.DataFrame({'feature': features, 'importance': importances})
feat_importance = feat_importance.sort_values(by='importance', ascending=False).head(20)

#visualisasi 
plt.figure(figsize=(10,6))
plt.barh(feat_importance['feature'], feat_importance['importance'])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance")
plt.show()
