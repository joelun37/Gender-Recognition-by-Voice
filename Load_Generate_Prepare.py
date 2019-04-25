# Import libraries and packages and functions
import librosa as lr
import librosa.display as lrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import re
from sklearn import preprocessing
from glob import glob
import wave
from scipy.io import wavfile
import csv
from Gender_Recognition_Functions import *

# Load audio file at 7kHz sampling rate
audio, sfreq = load_audio("/Volumes/750GB-HDD/CES-Data-Scientist/Data/06-11-22.wav", 7000)

# Generate Data for statistical learning:
# Short-Time Fourier Transform
# Power Spectogram
# Mel Spectogram
# MFCC
STFT_matrix, energy_power_spectogram, Mel_spectogram, MFCC = generate_data(audio, 175, 13, energy_power="power")

# Data preparation and cleaning on the generated data
x_, y_ = prepare_data(MFCC, audio, sfreq)

# Plotting Male & Female MFCC
male_idx = [i for i, y_ in enumerate(y) if y_ == 1]
male_mfcc = np.take(MFCC, male_idx, 1)
female_idx = [i for i, y_ in enumerate(y) if y_ == 0]
female_mfcc = np.take(MFCC, female_idx, 1)

plt.figure(figsize=(20, 4))

plt.subplot(1, 2, 1)
lrd.specshow(male_mfcc, y_axis='log')
plt.title('Male MFCC')
plt.colorbar()
plt.subplot(1, 2, 2)
lrd.specshow(female_mfcc, y_axis='log')
plt.title('Female MFCC')
plt.colorbar()

# Optionally, you can dump data to a csv file and try to run complex learning algorithms
# using online services such as Google Colab, on their GPUs

rounded_x_ = [ np.round(elem, 2) for elem in x_ ]

createCSV(rounded_x_, "/Volumes/750GB-HDD/CES-Data-Scientist/Output/MFCC.csv", multi_column = True)
createCSV(y_, "/Volumes/750GB-HDD/CES-Data-Scientist/Output/Response.csv", multi_column = True)