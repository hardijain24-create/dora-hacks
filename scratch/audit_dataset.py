import json
from collections import Counter

with open('public/models/isl-mvp/dataset.json') as f:
    data = json.load(f)

print(f'Total sequences: {len(data)}')
labels = [d['label'] for d in data]
print('Label counts:', Counter(labels))

seq = data[0]['sequence']
print(f'Sequence shape: {len(seq)} frames x {len(seq[0])} features')

sources = list(set(d.get('source', '?') for d in data))
print('Sources:', sources)
