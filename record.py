import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np

# Recording parameters
fs = 44100  # Sample rate
duration = 5  # seconds
filename = "output.wav"

print("Recording...")
# Record audio
recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Save the recorded audio to a WAV file
write(filename, fs, recording)

print(f"Playing back {filename}...")
# Load and play the recorded audio
samplerate, data = read(filename)
sd.play(data, samplerate)
sd.wait() # Wait until playback is finished
print("Playback finished.")