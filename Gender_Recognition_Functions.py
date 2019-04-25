# Import libraries and packages
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


# Load Audio File
def load_audio(audio_file, sfreq, mono = True):
    audio, sfreq = lr.load(audio_file, mono = mono, sr = sfreq)
    time = np.arange(0, len(audio)) / sfreq

    plt.figure()
    plt.subplot(3, 1, 1)
    lr.display.waveplot(audio, sr=sfreq)
    plt.title('Monophonic')
    print("Nombre de Samples : " + str(len(audio)))
    print("Sampling Rate : " + str(sfreq) + " Hz")
    print("Durée : " + str(len(audio) / (60 * sfreq)) + " min = " + str(len(audio) / (sfreq)) + " sec")

    return audio, sfreq
    
# Generate Envelope
def envelope(audio, rate, threshold):
    mask = []
    y = pd.Series(audio).apply(np.abs)
    y_mean = y.rolling(window = int(rate * 0.025), min_periods = 1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask

# Generate Data for statisttical learning:
# Short-Time Fourier Transform
# Power Spectogram
# Mel Spectogram
# MFCC
# Match MFCC data (explanatory variables) to gender of the speaker (explained variable)

def generate_data(audio, n_fft, n_mfcc, energy_power = "power"):
    # Computing the STFT matrix
    # Window length = 25 ms.
    # FFT window size = 7 000 * 0.025 = 175 samples
    # Using the default Hop Length = 175 / 4
    # Takes a couple of minutes
    STFT_matrix = lr.stft(audio, n_fft = n_fft)

    # Computing and displaying the energy or power spectogram
    if energy_power == "power":
        energy_power_spectogram = np.abs(STFT_matrix)**2
        lr.display.specshow(lr.amplitude_to_db(energy_power_spectogram, ref = np.max), y_axis = 'log', x_axis='time')
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        print(len(energy_power_spectogram))
    else:
        energy_power_spectogram = np.abs(STFT_matrix)
        lr.display.specshow(lr.amplitude_to_db(energy_power_spectogram, ref = np.max), y_axis = 'log', x_axis='time')
        plt.title('Energy spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        print(len(energy_power_spectogram))

    # Computing and displaying the Mel Spectogram
    Mel_spectogram = lr.feature.melspectrogram(S = energy_power_spectogram)

    plt.figure(figsize=(10, 4))
    lr.display.specshow(lr.power_to_db(Mel_spectogram, ref = np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB') 
    plt.title('Mel spectrogram')
    plt.tight_layout()

    # Computing and displaying Mel-frequency Cepstral Coefficients (MFCC)
    MFCC = lr.feature.mfcc(S=lr.power_to_db(Mel_spectogram), n_mfcc = n_mfcc)

    plt.figure(figsize=(10, 4))
    lr.display.specshow(MFCC, x_axis='s')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    return STFT_matrix, energy_power_spectogram, Mel_spectogram, MFCC
    
# Transpose MFCC
def prepare_data(MFCC, audio, sfreq, mask_threshold = 0.0005):
    """
    MFCC: MFCC array
    audio: audio sample loaded
    sfreq: sampling frequency of the audio sample
    mask_threshold : threshold for the sound envelope, default: 0.0005
    
    Return values: x_ for explanatory variables, y_ for response variable
    """
    x = np.transpose(MFCC)
    y = np.array(-1 * np.ones(len(x)))
    ratio = len(audio)/len(x)

    myPattern = r'      <Turn endTime="(.+)" speaker="(.+)" startTime="(.+)">'
    pattern = re.compile(myPattern)
    Male_Speaker = ["spk2","spk4"]
    mIdx = 0
    fIdx = 0
    f = open("/Volumes/750GB-HDD/CES-Data-Scientist/Data/06-11-22_manual.trs")
    for line in f:
        if "<Turn " in line:
            match = pattern.search(line)
            stop_inSec = match.group(1)
            speaker = match.group(2)
            start_inSec = match.group(3)
            start_inSample = float(start_inSec) * sfreq
            stop_inSample = float(stop_inSec) * sfreq
                
            mfcc_start_index = round(start_inSample / ratio)
            mfcc_stop_index = round(stop_inSample / ratio)
            
            if mfcc_stop_index > len(y):
                break
            elif (len(speaker) > 4) or (speaker == "spk0"):
                continue
            elif speaker in Male_Speaker:
                sex = 1
                mIdx = mIdx + (mfcc_stop_index - mfcc_start_index)
            else:
                sex = 0
                fIdx = fIdx + (mfcc_stop_index - mfcc_start_index)
            
            y[mfcc_start_index : mfcc_stop_index] = np.ones(mfcc_stop_index - mfcc_start_index)*sex
            
    # Cleaning
    # Noise and silence cancellation
    mask = envelope(audio, sfreq, threshold = mask_threshold)
    null_idx_x = [i for i, mask_ in enumerate(mask) if mask_ == False]
    null_idx_x_adjusted = null_idx_x//(np.ones(len(null_idx_x)) * ratio).astype(int)
    x_x = np.delete(x, null_idx_x_adjusted, 0)
    y_x = np.delete(y, null_idx_x_adjusted, 0)



    # Unmatched speaker
    null_idx = [i for i, y_ in enumerate(y_x) if y_ == -1]
    x_ = np.delete(x_x, null_idx, 0)
    y_ = np.delete(y_x, null_idx, 0)
    
    print("x: {}, y: {}, x_x: {}, y_x: {}, x_: {}, y_: {}".format(len(x), len(y), len(x_x), len(y_x), len(x_), len(y_)))

    return x_, y_

# Export X and Y for use on Google Colab
# Rounding to decrease of the CSV file

def createCSV(input, output_file, multi_column):
    """
    Function for creating a comma-delimited data file
    multi_column takes values True or False
    """
    csvfile = output_file
    if multi_column == True:
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator = '\n')
            writer.writerows(input)
    else:
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in input:
                writer.writerow([val])