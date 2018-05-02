# -*- coding: utf-8 -*-

# Created by Zuoqi Zhang on 2018/3/29.

import requests
import pandas as pd

df = pd.read_csv('/Users/saki/Downloads/data.csv')
df['ID'] = df['URL'].str.replace('https://open.spotify.com/track/', '')

data_us = df[df.Region == 'us'].drop(['URL', 'Region'], axis=1)
data_us.to_csv('data_us.csv')

# print(data_us)

# expires in 3600 secs
access_token = 'Bearer BQCGy9k7q50npWtt8uIWXu2FcEcVPOQJVbO0hZMeaOVN-xyWZuSobsCghzKQQzlYutb0aqSCH2x_mqr3B-s'
url = 'https://api.spotify.com/v1/audio-features/'

track_ids = data_us['ID'].unique()
tracks = []

for _id in track_ids:
    response = requests.get(url + _id, headers={'Authorization': access_token})
    if response.status_code != 200:
        break
    track = response.json()
    tracks.append(track)

tracks_df = pd.DataFrame(tracks).drop(['type', 'uri', 'track_href', 'analysis_url'], axis=1)
tracks_df.to_csv('tracks_df.csv')

features_df = pd.merge(data_us, tracks_df, how='left', left_on='ID', right_on='id')
features_df.to_csv('features_df.csv')