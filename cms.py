import numpy as np
import mmh3
import pandas as pd
import tqdm
import os

class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width))

    def add(self, flow_id, count=1):
        for i in range(self.depth):
            index = mmh3.hash(flow_id, i) % self.width
            self.table[i][index] += count

    def count(self, flow_id):
        min_count = float('inf')
        for i in range(self.depth):
            index = mmh3.hash(flow_id, i) % self.width
            min_count = min(min_count, self.table[i][index])
        return min_count

def get_features(cms, flow_id, actual_count):
    hash_counters = [cms.table[i][mmh3.hash(flow_id, i) % cms.width] for i in range(cms.depth)]
    estimated_count = cms.count(flow_id)

    # Constructing the feature dictionary
    features = {
        "Flow_ID": flow_id,
        "Actual_Count": actual_count,
        "Estimated_Count": estimated_count,
    }
    
    for i, hash_count in enumerate(hash_counters, start=1):
        features[f"Hash_Count_{i}"] = hash_count

    return features

def process_file(file_path):
    cms = CountMinSketch(7812, 4)
    actual_counts = {}
    chunksize = 1000000
    
    for chunk in pd.read_csv(file_path, sep='\t', header=None, chunksize=chunksize):
        for index, row in tqdm.tqdm(chunk.iterrows(), total=chunk.shape[0], desc=f"Processing lines in {os.path.basename(file_path)}"):
            flow_id = f'{row[0]}-{row[1]}'
            cms.add(flow_id)
            if flow_id in actual_counts:
                actual_counts[flow_id] += 1
            else:
                actual_counts[flow_id] = 1

    features = [get_features(cms, flow_id, actual_count) for flow_id, actual_count in actual_counts.items()]
    
    df_features = pd.DataFrame(features)
    
    # Construct CSV file path
    base_name = os.path.basename(file_path)
    csv_file_name = f'./{base_name}'
    df_features.to_csv(csv_file_name, index=False)

    return f"Processed and saved features to {csv_file_name}"

def main():
    file_path = ""
    process_file(file_path)