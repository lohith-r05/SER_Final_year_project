import librosa
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the audio
signal, sr = librosa.load(r'D:\SER_Project\first_voice.wav', sr=None)
time = np.linspace(0, len(signal)/sr, num=len(signal))

# Step 2: Plot time-domain signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (seconds)") 
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Framing the signal
frame_size = 0.025      # 25ms
frame_stride = 0.010    # 10ms(stride)
frame_length = int(round(frame_size * sr))
frame_step = int(round(frame_stride * sr))
signal_length = len(signal)

num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))   # 1D array, zeros for padding, length = pad_signal_length - original signal_length
pad_signal = np.append(signal, z)  # Padded signal
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
# 2D array, each row contains indices for a frame

frames = pad_signal[indices.astype(np.int32)] 

# Step 4: Plot a few original frames
selected_frames = [10, 50, 100]

plt.figure(figsize=(10, 6))
for i, frame_index in enumerate(selected_frames):
    plt.subplot(3, 1, i + 1)
    plt.plot(frames[frame_index])
    plt.title(f"Framed Signal {frame_index}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
plt.show()

# Step 5: Apply Hamming window
def hamming(N):
    return 0.54 - 0.46 * np.cos((2 * np.pi * np.arange(N)) / (N - 1))

hamm = hamming(frame_length)
windowed_frames = frames * hamm

# Step 6: FFT and Power Spectrum
NFFT = 512  # Number of FFT points

# Magnitude spectrum
mag_frames = np.absolute(np.fft.rfft(windowed_frames, NFFT))  # here rfft gives only positive frequencies

# Power spectrum
pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

# visualize the power spectrum
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(pow_frames.T), origin='lower', aspect='auto', cmap='jet',
           extent=[0, len(frames), 0, sr / 2])
plt.title('Spectrogram (Power Spectrum)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Frame Index')
plt.colorbar(label='Power (dB)')
plt.tight_layout()
plt.show()

# Trim silent parts of the signal
signal_trimmed, _ = librosa.effects.trim(signal)

# Step 7: MFCC Extraction
mfccs = librosa.feature.mfcc(y=signal_trimmed, sr=sr, n_mfcc=13)

# visualize MFCC
plt.figure(figsize=(10, 6))
plt.imshow(mfccs, origin='lower', aspect='auto', cmap='viridis')
plt.title('MFCC')
plt.xlabel('Frame Index')
plt.ylabel('MFCC Coefficients')
plt.colorbar()
plt.tight_layout()
plt.show()

np.set_printoptions(suppress=True, precision=6, floatmode='fixed') # to get better print format

# Print MFCC coefficients
print("MFCC Coefficients:")
print(mfccs)

# Step 8: Del MFCC and Del Del MFCC
delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)

print("\n Del MFCC Coefficients:") # Gives speed of change of speech features
print(delta_mfcc)

print("\n Del Del MFCC Coefficients:") # Gives acceleration of change of speech features
print(delta2_mfcc)

# Step 9: Combine into final 39 features
mfcc_features = np.vstack([mfccs, delta_mfcc, delta2_mfcc])

print("\nFinal 39-D Feature Matrix (MFCC + del MFCC + del del MFCC):")
print("Shape:", mfcc_features.shape)
print(mfcc_features)

# statistical features: mean and std for each of the 39 features
mfcc_mean = np.mean(mfcc_features, axis=1)
mfcc_std = np.std(mfcc_features, axis=1)

# Final fixed feature vector (78-D)
final_feature_vector = np.hstack([mfcc_mean, mfcc_std])

print("Final Feature Vector Shape:", final_feature_vector.shape)
print(final_feature_vector)

