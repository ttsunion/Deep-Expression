# -*- coding: utf-8 -*-
# fangshuming519@gmail.com
# /usr/bin/python3
import os
import wave
import numpy as np
from parameters import params as pm

def get_wav(sound_file):
    w = wave.open(os.path.join(wav_folder, sound_file),"rb")
    params = w.getparams()  
    #get the info  
    nchannels, sampwidth, framerate, nframes = params[:4]  
    #Reads and returns nframes of audio, as a string of bytes.   
    str_data = w.readframes(nframes)  
    #close the stream  
    w.close() 
    #turn the wave's data to array  
    wave_data = np.fromstring(str_data, dtype = np.short)/2**15
    if len(wave_data)>= pm.sr * pm.max_duration:
        wave_data = wave_data[:pm.sr * pm.max_duration]
    else:
        wave_data = np.concatenate((wave_data, np.array([0] * (int(pm.sr * pm.max_duration) - len(wave_data)))))
    wave_data = wave_data.reshape(pm.Dy, pm.Ty)
    return wave_data	

if __name__ == "__main__":
    wav_folder = os.path.join('samples', 'wavs')
    wav_processed_folder  = os.path.join('processed', 'wavs')
    #audio processing
    files = os.listdir(wav_folder)
    for f in files:
        wave_data = get_wav(f)
        np.save(os.path.join(wav_processed_folder, f.replace(".wav", ".npy")), wave_data)
