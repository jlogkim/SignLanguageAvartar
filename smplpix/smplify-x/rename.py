import os

file_path = '/home/hyunsoo/Dev/GitHub/smplpix/smplify-x/data/keypoints'
file_names = sorted(os.listdir(file_path))
i = 0
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i+1).zfill(5) + '_keypoints.json'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1