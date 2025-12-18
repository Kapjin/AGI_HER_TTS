import numpy as np

dur = np.load("preprocessed_data/KSS/duration/1-duration-1_0000.npy")
mel = np.load("preprocessed_data/KSS/mel/1-mel-1_0000.npy")
pitch = np.load("preprocessed_data/KSS/pitch/1-pitch-1_0000.npy")
energy = np.load("preprocessed_data/KSS/energy/1-energy-1_0000.npy")

print("duration len:", len(dur))           # phoneme 개수
print("mel length:", mel.shape[0])         # frame 개수
print("pitch length:", len(pitch))
print("energy length:", len(energy))
print("sum(duration):", dur.sum())
