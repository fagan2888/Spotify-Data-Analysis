# -*- coding: utf-8 -*-

# Created by Zuoqi Zhang on 2018/3/30.

import io
import numpy as np
import pandas as pd
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier

features = ['acousticness',
            'danceability',
            'duration_ms',
            'energy',
            'instrumentalness',
            'key',
            'liveness',
            'loudness',
            'mode',
            'speechiness',
            'tempo',
            'time_signature',
            'valence']

tracks_df = pd.read_csv('tracks_df.csv')
data_us = pd.read_csv('data_us.csv')

label_df = data_us[['ID', 'Streams']]
label_df = label_df.groupby('ID').sum().reset_index()
label_df['label'] = np.where(label_df['Streams'] >= label_df['Streams'].mean(), 1, 0)

data = pd.merge(tracks_df, label_df, how='left', left_on='id', right_on='ID')
data = data[features + ['label']]
data.to_csv('dataset.csv')

train, test = train_test_split(data, test_size=0.1, shuffle=True)
X_train = train[features]
y_train = train['label']
X_test = test[features]
y_test = test['label']

clf = DecisionTreeClassifier()
dt = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using Decision Tree: {}%'.format(round(score, 1)))

# f = io.StringIO()
# export_graphviz(dt, out_file=f, feature_names=features)
# graph_from_dot_data(f.getvalue()).write_png('Decision Tree.png')

clf = KNeighborsClassifier(10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy using KNN: {}%'.format(round(score, 1)))