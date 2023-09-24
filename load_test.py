import time
from pathlib import Path

import pandas as pd

from utils import text_preprocessing

if __name__ == "__main__":
    data, tf_idfs = text_preprocessing()
    df = pd.DataFrame(tf_idfs).fillna(0)
    filtered_df = []
    filtered_df_index = []
    public_figure = ['anies', 'ganjar', 'prabowo', 'puan', 'ridwan', 'ahok', 'kamil', 'ridw']
    for d in df.index:
        isin_most5words = df.T[d].sort_values(ascending=False).index[0:5].isin(public_figure).any()
        if isin_most5words:
            filtered_df.append(df.T[d].to_numpy())
            filtered_df_index.append(d)

    df_filtered = pd.DataFrame(filtered_df, index=filtered_df_index, columns=df.columns)

    # Prepare Path
    folder_name = time.time()
    folder_path = './data/test'.format(folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    df_filtered.to_pickle('{}/{}.pkl.xz'.format(folder_path, time.time()))
