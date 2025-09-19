import librosa
import numpy as np
import joblib

# Load audio
signal, sr = librosa.load(r"TESS_speech/TESS Toronto emotional speech set data/YAF_neutral/YAF_bone_neutral.wav", sr=None)

# Trim silence
signal_trimmed, _ = librosa.effects.trim(signal)

# Frame parameters
frame_size = 0.025
frame_stride = 0.010
frame_length = int(round(frame_size * sr))
frame_step = int(round(frame_stride * sr))
signal_length = len(signal_trimmed)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(signal_trimmed, z)

# Compute indices and frames
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32)]

# Apply Hamming window
hamm = 0.54 - 0.46 * np.cos((2 * np.pi * np.arange(frame_length)) / (frame_length - 1))
windowed_frames = frames * hamm

# MFCC features (13 MFCC + delta + delta-delta = 39)
mfccs = librosa.feature.mfcc(y=signal_trimmed, sr=sr, n_mfcc=13, n_fft=512,
                             hop_length=frame_step, n_mels=26, fmax=sr/2)
delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)
mfcc_feat = np.vstack([mfccs, delta_mfcc, delta2_mfcc])  # 39 x frames

# Additional spectral features
energy = np.sum(frames**2, axis=1, keepdims=True).T
zcr = librosa.feature.zero_crossing_rate(signal_trimmed, frame_length=frame_length, hop_length=frame_step)
spec_cent = librosa.feature.spectral_centroid(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)
spec_bw = librosa.feature.spectral_bandwidth(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)
spec_roll = librosa.feature.spectral_rolloff(y=signal_trimmed, sr=sr, n_fft=512, hop_length=frame_step)

# Align all features
min_frames = min(mfcc_feat.shape[1], energy.shape[1], zcr.shape[1],
                 spec_cent.shape[1], spec_bw.shape[1], spec_roll.shape[1])
mfcc_feat = mfcc_feat[:, :min_frames]
energy = energy[:, :min_frames]
zcr = zcr[:, :min_frames]
spec_cent = spec_cent[:, :min_frames]
spec_bw = spec_bw[:, :min_frames]
spec_roll = spec_roll[:, :min_frames]

# Combine additional features
additional_feats = np.vstack([energy, zcr, spec_cent, spec_bw, spec_roll])  # 5 x frames

# Combine MFCC + additional features
all_feats = np.vstack([mfcc_feat, additional_feats])  # 44 x frames

# Statistical features (mean & std for each row)
feat_mean = np.mean(all_feats, axis=1)
feat_std = np.std(all_feats, axis=1)

# Delta and Delta-Delta of additional features only
delta_add = librosa.feature.delta(additional_feats)
delta2_add = librosa.feature.delta(additional_feats, order=2)

# Final 108-dimensional feature vector
final_feature_vector = np.hstack([
    feat_mean, feat_std,
    np.mean(delta_add, axis=1), np.std(delta_add, axis=1),
    np.mean(delta2_add, axis=1), np.std(delta2_add, axis=1)
]).reshape(1, -1)  # shape (1, 108)

print("Final Feature Vector Shape:", final_feature_vector.shape)

# Load trained model
scaler, model = joblib.load("ser_svm_best_5emo_final.pkl")

final_feature_vector = scaler.transform(final_feature_vector)

# Predict
prediction = model.predict(final_feature_vector)[0]

# Map emotion labels
emotion_map = {
    0:'neutral',
    1:'happy',
    2:'sad',
    3:'angry',
    4:'fearful'
}

print("Predicted Emotion:", emotion_map.get(prediction, "Unknown"))