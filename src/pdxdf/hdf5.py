import pandas as pd

def save(path, df, key="stream"):
    if "stream_info" not in df.attrs:
        raise ValueError("DataFrame missing attribute: stream_info.")
    info = df.attrs["stream_info"]
    if (df.dtypes != info.channel_format).any():
        df = df.astype(info.channel_format)
    with pd.HDFStore(path) as store:
        store.put(key, df, format="table")
        store.get_storer(key).attrs.stream_info = info


def load(path, key="stream", start=0, stop=None):
    with pd.HDFStore(path, mode="r") as store:
        df = pd.read_hdf(store, key, start=start, stop=stop)
        info = store.get_storer(key).attrs.stream_info
        df.attrs["stream_info"] = info
    return df


def load_stream_info(path, key="stream"):
    with pd.HDFStore(path, mode="r") as store:
        info = store.get_storer(key).attrs.stream_info
        return info
