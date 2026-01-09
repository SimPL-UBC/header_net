import sys
import os

path1 = '/home/aerial/repository/heimdall_net/1st_place_kaggle_player_contact_detection/cnn/models'
path2 = '/home/aerial/repository/heimdall_net/1st_place_kaggle_player_contact_detection/cnn'

print(f"Adding {path1}")
sys.path.append(path1)
print(f"Adding {path2}")
sys.path.append(path2)

print(f"sys.path: {sys.path}")

try:
    import resnet3d_csn
    print("Import successful")
    print(resnet3d_csn)
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
