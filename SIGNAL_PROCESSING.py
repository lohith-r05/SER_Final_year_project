import librosa
import numpy as np
import pandas as pd

# Step 1: Load audio
signal, sr = librosa.load(r'D:\SER_Project\first_voice.wav', sr=None)

# Step 2: Trim silence
signal_trimmed, _ = librosa.effects.trim(signal)

# Step 3: Frame parameters
frame_size = 0.025
frame_stride = 0.010
frame_length = int(round(frame_size * sr))
frame_step = int(round(frame_stride * sr))
signal_length = len(signal_trimmed)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(signal_trimmed, z)

# Step 4: Compute indices and frames
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32)]

# Step 5: Apply Hamming window
hamm = 0.54 - 0.46 * np.cos((2 * np.pi * np.arange(frame_length)) / (frame_length - 1))
windowed_frames = frames * hamm

# Step 6: MFCC features (13 MFCC + delta + delta-delta = 39)
mfccs = librosa.feature.mfcc(y=signal_trimmed, sr=sr, n_mfcc=13, n_fft=512, hop_length=frame_step, n_mels=26, fmax=sr/2)
delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)
mfcc_feat = np.vstack([mfccs, delta_mfcc, delta2_mfcc])  # 39 x frames

# Step 7: Additional spectral features (energy, ZCR, spectral centroid, bandwidth, roll-off)
energy = np.sum(frames**2, axis=1, keepdims=True).T
zcr = librosa.feature.zero_crossing_rate(signal_trimmed, frame_length=frame_length, hop_length=frame_step)
spec_cent = librosa.feature.spectral_centroid(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)
spec_bw = librosa.feature.spectral_bandwidth(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)
spec_roll = librosa.feature.spectral_rolloff(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)

# Step 8: Align all features to same number of frames
min_frames = min(mfcc_feat.shape[1], energy.shape[1], zcr.shape[1], spec_cent.shape[1], spec_bw.shape[1], spec_roll.shape[1])
mfcc_feat = mfcc_feat[:, :min_frames]
energy = energy[:, :min_frames]
zcr = zcr[:, :min_frames]
spec_cent = spec_cent[:, :min_frames]
spec_bw = spec_bw[:, :min_frames]
spec_roll = spec_roll[:, :min_frames]

# Step 9: Combine additional features
additional_feats = np.vstack([energy, zcr, spec_cent, spec_bw, spec_roll])  # 5 x frames

# Step 10: Combine MFCC + additional features
all_feats = np.vstack([mfcc_feat, additional_feats])  # 44 x frames

# Step 11: Compute statistical features (mean & std for each row)
feat_mean = np.mean(all_feats, axis=1)
feat_std = np.std(all_feats, axis=1)

# Step 12: Delta and Delta-Delta of additional features only
delta_add = librosa.feature.delta(additional_feats)
delta2_add = librosa.feature.delta(additional_feats, order=2)

# Step 13: Final 108-dimensional feature vector
final_feature_vector = np.hstack([
    feat_mean, feat_std,
    np.mean(delta_add, axis=1), np.std(delta_add, axis=1),
    np.mean(delta2_add, axis=1), np.std(delta2_add, axis=1)
])  # 108D vector

# Step 14: Print final feature vector
np.set_printoptions(suppress=True, precision=2)
print("Final Feature Vector Shape (108D):", final_feature_vector.shape)
print(final_feature_vector)

# Step 15: Save to CSV
pd.DataFrame([final_feature_vector]).to_csv("features.csv", index=False)