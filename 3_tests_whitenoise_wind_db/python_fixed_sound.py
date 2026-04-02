import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
DURATION = 30
#FREQUENCY = 261.63  # Middle C (C4)
FREQUENCY = 32.70 #Hz lowest C
#FREQUENCY = 1046.50 #Hz highest C (C6)

t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
tone = np.sin(2 * np.pi * FREQUENCY * t).astype(np.float32)

sd.play(tone, SAMPLE_RATE)
sd.wait()