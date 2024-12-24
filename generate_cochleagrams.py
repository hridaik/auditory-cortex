import os
import librosa
import numpy as np
import scipy.ndimage

# Parameters for cochleagram generation
sampling_rate = 16000 # Hz
n_fft = 1024 # FFT window
hop_length = 512 # 50% overlap between FFT windows
n_mels = 128 # Frequency bins
output_dir = "C:/Users/hridai/Desktop/DONN/datasets/cochleagrams"

def process_audio_to_cochleagram(audio_path):
    """Converts a given audio file to a cochleagram."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels)

    # Convert to decibels
    cochleagram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Optional: Gaussian smoothing
    cochleagram = scipy.ndimage.gaussian_filter(cochleagram, sigma=1)

    return cochleagram

def save_cochleagram(cochleagram, save_path):
    """Saves the cochleagram as a NumPy file."""
    np.save(save_path, cochleagram)

def process_dataset(dataset_dir):
    """Processes the dataset and generates cochleagrams for all audio files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for speaker in os.listdir(dataset_dir):
        speaker_path = os.path.join(dataset_dir, speaker)

        if os.path.isdir(speaker_path):
            # Create output directory for the speaker
            speaker_output_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_output_dir, exist_ok=True)

            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(speaker_path, audio_file)
                    save_path = os.path.join(speaker_output_dir, audio_file.replace(".wav", ".npy"))

                    # Generate and save cochleagram
                    cochleagram = process_audio_to_cochleagram(audio_path)
                    save_cochleagram(cochleagram, save_path)

                    print(f"Processed {audio_file} and saved to {save_path}")

# Example usage
dataset_dir = "C:/Users/hridai/Desktop/DONN/datasets/16000_pcm_speeches/"
process_dataset(dataset_dir)


# audio_file = "C:/Users/hridai/Desktop/DONN/datasets/16000_pcm_speeches/Nelson_Mandela/0.wav"
# audio_file = "C:/Users/hridai/Desktop/DONN/datasets/16000_pcm_speeches/_background_noise_/10convert.com_Audience-Claps_daSG5fwdA7o.wav"
# cgm = process_audio_to_cochleagram(audio_file)
# save_cochleagram(cgm, 'C:/Users/hridai/Desktop/DONN/datasets/test_bg.npy')



# import matplotlib.pyplot as plt
# import librosa.display

# def plot_cochleagram(cochleagram, title="Cochleagram"):
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(cochleagram, sr=16000, hop_length=512, 
#                              x_axis='time', y_axis='mel', cmap='magma')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

# # Load a cochleagram for visualization
# cochleagram = np.load('C:/Users/hridai/Desktop/DONN/datasets/test_bg.npy')
# plot_cochleagram(cochleagram, title="Example Cochleagram")

