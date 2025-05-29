import pyarrow as pa

def read_arrow_to_df(path):
    with open(path, "rb") as f:
        reader = pa.ipc.RecordBatchStreamReader(f)
        df = reader.read_pandas()
        return df

path = 'datasets/dpo_solidity_data/test/data-00000-of-00001.arrow'
# path = 'datasets/dpo_solidity_data/train/data-00000-of-00001.arrow'

df = read_arrow_to_df(path)
print(df)