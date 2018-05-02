import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_us.csv')
featuresdf = pd.read_csv('features_df.csv')

artists = df[['Artist', 'Streams']]
artists = artists.groupby('Artist')
artists = artists.sum().sort_values('Streams', ascending=False)
artists = artists.reset_index()
artists.index = np.arange(1, len(artists) + 1)
artists = artists.head(10)        # top 10
# print(artists)

tracks = df[['Track Name', 'Streams']]
tracks = tracks.groupby('Track Name')
tracks = tracks.sum().sort_values('Streams', ascending=False)
tracks = tracks.reset_index()
tracks.index = np.arange(1, len(tracks) + 1)
tracks = tracks.head(100)        # top 100
# print(tracks)

top_features = featuresdf.loc[featuresdf['name'].isin(tracks['Track Name'])]
# print(top_features)

features = top_features.drop(['id', 'name', 'artists'], axis=1)

# correlations = features.corr()
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, 13, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(features, fontsize=8, rotation=90)
# ax.set_yticklabels(features, fontsize=8)
# plt.tight_layout()
# plt.show()

n = 50
df2 = pd.DataFrame({'x': range(1, n + 1),
                 'danceability': top_features['danceability'].head(n),
                 'energy': top_features['energy'].head(n),
                 'speechiness': top_features['speechiness'].head(n),
                 'acousticness': top_features['acousticness'].head(n),
                 'liveness': top_features['liveness'].head(n),
                 'valence': top_features['valence'].head(n)})

palette = plt.get_cmap('Set1')

num = 0
for column in df2.drop('x', axis=1):
    num += 1

    plt.subplot(3, 3, num)

    plt.plot(df2['x'], df2[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

    plt.xlim(0, 30)
    plt.ylim(0, 1)

    if num in range(7):
        plt.tick_params(labelbottom='off')
    if num not in [1, 4, 7]:
        plt.tick_params(labelleft='off')

    plt.title(column, loc='left', fontsize=8, fontweight=0, color=palette(num))
plt.show()