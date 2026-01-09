import numpy as np
import pandas as pd
import os

# Create directory if not exists
os.makedirs('cache/cache_header', exist_ok=True)

# Create dummy npy file
# Shape: (16, 224, 224, 3) - assuming 16 frames, 224x224, RGB
dummy_data = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
dummy_path = 'cache/cache_header/dummy_sample_s.npy'
np.save(dummy_path, dummy_data)
print(f"Created {dummy_path}")

# Create dummy CSV
df = pd.DataFrame({
    'path': ['cache/cache_header/dummy_sample'] * 10, # 10 samples
    'label': [0, 1] * 5,
    'video_id': ['dummy_video'] * 10,
    'half': [1] * 10,
    'frame': [100] * 10
})

csv_path = 'cache/cache_header/dummy_train.csv'
df.to_csv(csv_path, index=False)
print(f"Created {csv_path}")
