import numpy as np
import scipy as sp
import os
import soundfile as sf
import pandas as pd

def next_power_2(x):
    n = 2
    y = x - 1
    while y < x:
        y = 2**n
        n = n + 1
    return y

def direct_sound_index(h):
    assert h.ndim == 1, "h array contains more than one dimension"
    h_max = np.abs(h).max()
    idx = np.where(np.abs(h) > h_max * 0.8)[0]
    return int(idx[0])

#-----------------------------
# === Gammatone filter bank ===
#----------------------------- 
# Implementation heavily insired by V. Hohmann. Frequency analysis and synthesis using a gammatone filterbank. Acta Acustica united with Acoustica, 88(3):433--442, 2002.
#  and the implementation 
class gammatone_filterbank():
    def __init__(self, Fs):
        self.Fs = Fs
        self.N = 42 # Number of gammatone filter bands
        self.lower_cutoff = 20
        self.upper_cutoff = Fs//2
        self.specified_center_frequency_hz = 20
        self.gamma_order = 8
        self.bandwidth_factor = 1
        self.filters_per_ERB = 1
        
        self.CF = self.center_frequencies()
        # Precompute filters
        self.filters = [self.gammatone_filter(cf) for cf in self.CF]
        
    def filter(self, h):
        """Returns FIR gammatone filter bank channels of input signal"""
        h_GFB = np.zeros([len(h), self.N])
       
        for n in range(self.N):
            h_GFB[:,n] = sp.signal.fftconvolve(h, self.filters[n], mode='same')
        return h_GFB
    
    def gammatone_filter(self, center_freq):        
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        
        filter_length = int(np.ceil(4 / (2 * np.pi * erb) * self.Fs))
        
        t = np.arange(filter_length) / self.Fs
        
        g = t**(self.gamma_order-1) * np.exp(-2*np.pi*erb*t) * np.cos(2*np.pi*center_freq*t)

        G = np.fft.fft(g, 2*len(g))
        # Normalize so that max amplitude is 1
        g /= np.max(np.abs(G))
        
        return g

    def center_frequencies(self):
        # Convert frequencies to the ERB scale
        lower_cut_ERB = Hz_2_ERB(self.lower_cutoff)
        specified_center_frequency_ERB = Hz_2_ERB(self.specified_center_frequency_hz)
        upper_cut_ERB = Hz_2_ERB(self.upper_cutoff)
        
        # Calculate the number of filters below the base frequency
        erbs_below_base_frequency = specified_center_frequency_ERB - lower_cut_ERB
        num_of_filters_below_base_freq = np.floor(erbs_below_base_frequency * self.filters_per_ERB)
        
        # Calculate center frequency of lowest band
        start_frequency_erb = specified_center_frequency_ERB - num_of_filters_below_base_freq / self.filters_per_ERB
        
        # Create vector of all center frequencies
        center_frequencies_erb = np.arange(start_frequency_erb, upper_cut_ERB, 1 / self.filters_per_ERB)
        return ERB_2_Hz(center_frequencies_erb)
        
def Hz_2_ERB(Hz):
    L = 24.7
    Q = 9.265
    return Q * np.log(1 + Hz / (L * Q))

def ERB_2_Hz(ERBscale):
    L = 24.7
    Q = 9.265
    return (np.exp(ERBscale / Q) - 1) * (L * Q)


def read_metadata(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return pd.read_csv(filename)
    else:
        raise ValueError("read_metadata: CSV filename does not exist")
    
    
def read_simulation_data(data_folder, id):
    array_folder = data_folder + "/rirs/"
    h_arr, Fs = sf.read(os.path.join(array_folder, f"rirs_{id}_h_opt_eval.wav"))
    d_arr, Fs = sf.read(os.path.join(array_folder, f"rirs_{id}_d_eval.wav"))
    
    binaural_folder = data_folder + "/brirs/"
    h_bin, Fs = sf.read(os.path.join(binaural_folder, f"brirs_{id}_h_opt_b.wav"))
    d_bin, Fs = sf.read(os.path.join(binaural_folder, f"brirs_{id}_d_b.wav"))
    
    dr_folder = data_folder + "/drr/"
    h_dr, Fs = load_pair(dr_folder, id, "h_opt_eval")
    d_dr, Fs = load_pair(dr_folder, id, "d_eval")
    h_dr_bin, Fs = load_pair(dr_folder, id, "h_opt_brir")
    d_dr_bin, Fs = load_pair(dr_folder, id, "d_brir")
    
    return h_arr, h_bin, h_dr,h_dr_bin,  d_arr, d_bin, d_dr, d_dr_bin


def load_pair(path_dir, id, prefix):
    direct, Fs = sf.read(os.path.join(path_dir, f"{prefix}_{id}_dir.wav"))
    reverb, Fs = sf.read(os.path.join(path_dir, f"{prefix}_{id}_rev.wav"))
    out = np.stack((direct, reverb), axis=0)
    return out, Fs

def create_results_dataframe(metadata_dataframe, index):
    row = metadata_dataframe.loc[metadata_dataframe["id"] == index].iloc[0]
    if row["N_sources"] == 2:
        row["N_sources"] = 3
    results_dataframe = pd.DataFrame([{
    "id": row["id"],
    "N_sources": row["N_sources"],
    "N_receivers": row["N_receivers"],
    "method": row["method"],
    "delay_supporting": row['delay_supporting'],
    "ommit_primary": row['ommit_primary'],
    "filter_length": row["filter_length"],
    "Regularisation_weight": row["Regularisation weight"],
    "FIR_shaping_weight": row["filter shaping weight"],
    "NFFT": row["NFFT"],
    "target_room": row["target_room"],
    "playback_room": row["playback_room"],
    "Receiver_type": None,
    "Receiver_index": None,
    "Spectral_RMSE": None,
    "IACC_RMSE": None,
    "DRR_RMSE": None,
    "EDC_RMSE": None,
    "C50_E": None,
    "C50_GFB_RMSE": None,
    }])
    
    return results_dataframe

def append_dataframe_to_csv(filename, df_new):
    # Check if file exists and is non-empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df_existing = pd.read_csv(filename)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new.copy()
    
    # Save back to CSV
    df_updated.to_csv(filename, index=False)