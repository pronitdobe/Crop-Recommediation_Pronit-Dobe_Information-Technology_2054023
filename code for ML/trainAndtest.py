import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score  # Import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib 
import pickle

df = pd.read_csv('Crop_recommendation.csv')
df.head()

x = df.drop('label', axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

model_4 = RandomForestClassifier(n_estimators=25, random_state=2)
model_4.fit(x_train.values, y_train.values)

y_pred_4 = model_4.predict(x_test)
random_fore_acc = accuracy_score(y_test, y_pred_4)
print("Accuracy of Random Forest is " + str(random_fore_acc))

file_name = 'crop_app'
joblib.dump(model_4, 'crop_app')

app = joblib.load('crop_app')

Pkl_Filename = "Pickle_RL_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model_4, file)

with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

Pickled_Model
