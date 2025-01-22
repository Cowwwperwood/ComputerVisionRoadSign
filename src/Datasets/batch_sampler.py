import random
import torch
from torch.utils.data import Sampler
from tqdm import tqdm
from collections import defaultdict

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, elems_per_class, classes_per_batch, use_tqdm=True):
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.use_tqdm = use_tqdm
        
        self.class_indices = defaultdict(list)
        
        if self.use_tqdm:
            print("Building class indices...")
        
        for idx, (_, _, label) in tqdm(enumerate(data_source), total=len(data_source), disable=not self.use_tqdm, desc="Building indices"):
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())

    def __iter__(self):
        batch_indices = []
        
        for _ in tqdm(range(len(self)), desc="Generating batches", disable=not self.use_tqdm):
            selected_classes = random.sample(self.classes, self.classes_per_batch)
            
            batch = []
            for c in selected_classes:
                class_samples = random.sample(self.class_indices[c], self.elems_per_class)
                batch.extend(class_samples)
            
            random.shuffle(batch)
            batch_indices.append(batch)
            
            yield torch.tensor(batch)

    def __len__(self):
        return len(self.data_source) // (self.elems_per_class * self.classes_per_batch)