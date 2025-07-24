import pandas as pd

def save(path, df):
    if "stream_info" not in df.attrs:
        raise ValueError("DataFrame missing attribute: stream_info.")
    info = df.attrs["stream_info"]
    if (df.dtypes != info.channel_format).any():
        df = df.astype(info.channel_format)
    with pd.HDFStore(path) as store:
        store.put("stream", df, format="table")
        store.get_storer("stream").attrs.stream_info = info


def load(path, start=0, stop=None):
    with pd.HDFStore(path, mode="r") as store:
        df = pd.read_hdf(store, "stream", start=start, stop=stop)
        info = store.get_storer("stream").attrs.stream_info
        df.attrs["stream_info"] = info
    return df


def load_stream_info(path):
    with pd.HDFStore(path, mode="r") as store:
        info = store.get_storer("stream").attrs.stream_info
        return info
