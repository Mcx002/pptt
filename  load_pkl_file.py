import pandas as pd

pickle = "/home/damian/research/python/preprocessing-text/data/saved/1695122312.6980379.pkl.xz"


if __name__ == "__main__":
    df = pd.read_pickle(pickle)
    print(df[0]['cluster'])