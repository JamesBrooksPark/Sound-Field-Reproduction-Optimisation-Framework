import numpy as np
import os
from src.numerical_evaluation.numerical_helper import next_power_2, direct_sound_index, gammatone_filterbank, read_metadata, read_simulation_data, create_results_dataframe, append_dataframe_to_csv

from src.ComplexSmoothing import Smoothing

class NumericalMeasures():
    def __init__(self, Fs):
        # File paths
        self.meta_data_path = None
        self.data_path = None
        self.results_data_path = None
        
        # Parameters        
        self.Fs = Fs
        self.GFB = gammatone_filterbank(self.Fs)
        
        # RMSE
        self.RMSE_smoothing = True
        self.RMSE_smoothing_octave = 3
        self.RMSE_upper_Hz = 20000
        
        # IACC
        
        # DRR
        self.DRR_smoothing = True
        self.DRR_smoothing_octave = 3
        self.DRR_upper_Hz = 20000
        
        # EDC

        
    def Spectral_RMSE(self,h,d):
        assert h.ndim == 1 , "spectral_deviation: h array contains more than one dimension"
        assert d.ndim == 1 , "spectral_deviation: d array contains more than one dimension"
        assert self.RMSE_upper_Hz <= self.Fs / 2
        
        NFFT = next_power_2(max(len(h),len(d)))
        
        upper_bin = (self.RMSE_upper_Hz/self.Fs) * NFFT
            
        if self.RMSE_smoothing:
            Sm = Smoothing(NFFT)
            h = Smoothing.complexSmoothing(Sm, h, "spectrum", 1, 1 / self.RMSE_smoothing_octave)
            d = Smoothing.complexSmoothing(Sm, d, "spectrum", 1, 1 / self.RMSE_smoothing_octave)
        
        H = np.abs(np.fft.rfft(h, self.NFFT))[:upper_bin]
        D = np.abs(np.fft.rfft(d, self.NFFT))[:upper_bin]
        
        RMSE = np.sqrt(np.mean((20*np.log10(H/D))**2))
        
        return RMSE
    
    
    def IACC_RMSE(self,h,d):
        assert h.ndim == 2 and h.shape[1] == 2, "IACC_deviation: h array does not have shape J x 2"
        assert d.ndim == 2 and d.shape[1] == 2, "IACC_deviation: d array does not have shape J x 2"
        
        h_GFB_L = self.GFB.filter(h[:,0])
        h_GFB_R = self.GFB.filter(h[:,1])
        d_GFB_L = self.GFB.filter(d[:,0])
        d_GFB_R = self.GFB.filter(d[:,1])
        
        h_IACC = np.sum(h_GFB_L * h_GFB_R, axis=0) / np.sqrt(np.sum(h_GFB_L**2, axis=0) * np.sum(h_GFB_R**2, axis=0))
        d_IACC = np.sum(d_GFB_L * d_GFB_R, axis=0) / np.sqrt(np.sum(d_GFB_L**2, axis=0) * np.sum(d_GFB_R**2, axis=0))
        
        return np.sqrt(np.mean((h_IACC - d_IACC)**2))
    
    
    def DRR_RMSE(self, h_dir, h_rev, d_dir, d_rev):
        assert h_dir.ndim == 1, "DRR_deviation: h_dir array contains more than one dimension"
        assert h_rev.ndim == 1, "DRR_deviation: h_rev array contains more than one dimension"
        assert d_dir.ndim == 1, "DRR_deviation: d_dir array contains more than one dimension"
        assert d_rev.ndim == 1, "DRR_deviation: d_rev array contains more than one dimension"
        assert self.DRR_upper_Hz <= self.Fs / 2
        
        NFFT = next_power_2(max(len(h_dir),len(d_dir)))
        
        upper_bin = (self.RMSE_upper_Hz/self.Fs) * NFFT

        if self.DRR_smoothing:
            Sm = Smoothing(self.NFFT)
            h_dir = Smoothing.complexSmoothing(Sm, h_dir, "spectrum", 1, 1 / self.DRR_smoothing_octave)
            h_rev = Smoothing.complexSmoothing(Sm, h_rev, "spectrum", 1, 1 / self.DRR_smoothing_octave)
            d_dir = Smoothing.complexSmoothing(Sm, d_dir, "spectrum", 1, 1 / self.DRR_smoothing_octave)
            d_rev = Smoothing.complexSmoothing(Sm, d_rev, "spectrum", 1, 1 / self.DRR_smoothing_octave)

        H_dir = np.abs(np.fft.rfft(h_dir, NFFT))[:upper_bin]
        H_rev = np.abs(np.fft.rfft(h_rev, NFFT))[:upper_bin]
        D_dir = np.abs(np.fft.rfft(d_dir, NFFT))[:upper_bin]
        D_rev = np.abs(np.fft.rfft(d_rev, NFFT))[:upper_bin]

        eps = 1e-10
        H_DRR_dB = 10 * np.log10((H_dir**2 + eps) / (H_rev**2 + eps))
        D_DRR_dB = 10 * np.log10((D_dir**2 + eps) / (D_rev**2 + eps))

        return np.sqrt(np.mean((H_DRR_dB - D_DRR_dB)**2))
    
    
    def EDC_RMSE(self, h, d):
        assert h.ndim == 1, "h array must be 1D"
        assert d.ndim == 1, "d array must be 1D"
        
        h_edc = np.cumsum(h[::-1]**2)[::-1]
        d_edc = np.cumsum(d[::-1]**2)[::-1]
        

        h_edc /= (h_edc[0] + 1e-12)
        d_edc /= (d_edc[0] + 1e-12)
        
        eps = 1e-12
        h_edc_db = 10 * np.log10(h_edc + eps)
        d_edc_db = 10 * np.log10(d_edc + eps)
        
        return np.sqrt(np.mean((h_edc_db - d_edc_db)**2))
    
    
    def C50_E(self,h,d):
        h_direct_index = direct_sound_index(h)
        d_direct_index = direct_sound_index(d)

        N50 = int(self.Fs * 0.05)
        
        h_c50 = 10*np.log10(sum(h[h_direct_index:h_direct_index+N50]**2) / sum(h[h_direct_index+N50:]**2))
        d_c50 = 10*np.log10(sum(d[d_direct_index:d_direct_index+N50]**2) / sum(d[d_direct_index+N50:]**2))
    
        return abs(h_c50 - d_c50)
    
    
    def C50_GFB_RMSE(self,h,d):
        h_direct_index = direct_sound_index(h)
        d_direct_index = direct_sound_index(d)
        
        h_GFB = self.GFB.filter(h)
        d_GFB = self.GFB.filter(d)

        N50 = int(self.Fs * 0.05)
        
        h_c50 = 10*np.log10(np.sum(h_GFB[h_direct_index:h_direct_index+N50]**2, axis=0) / np.sum(h_GFB[h_direct_index+N50:]**2, axis=0))
        d_c50 = 10*np.log10(np.sum(d_GFB[d_direct_index:d_direct_index+N50]**2, axis=0) / np.sum(d_GFB[d_direct_index+N50:]**2, axis=0))
    
        return np.sqrt(np.mean((h_c50 - d_c50)**2))
    
    
    def run_measures(self):
        meta_data = read_metadata(self.meta_data_path)
        N =  meta_data["id"].max() +1
        
        if os.path.exists(self.results_data_path):
            os.remove(self.results_data_path)
            
        for n in range(N):
            h_arr, h_bin, h_dr, h_dr_bin, d_arr, d_bin, d_dr, d_dr_bin = read_simulation_data(n)
            
            results_dataframe = create_results_dataframe(meta_data, n)
            
            # Array measures
            M = np.shape(h_arr)[1]
            for m in range(M):
                results_dataframe_m = results_dataframe
                results_dataframe_m.loc[0, "Receiver_type"] = "Array"
                results_dataframe_m.loc[0, "Receiver"] = m
                results_dataframe_m.loc[0, "Spectral_RMSE"] = self.Spectral_RMSE(h_arr[:,m], d_arr[:,m])
                results_dataframe_m.loc[0, "IACC_RMSE"] = float('NaN')
                results_dataframe_m.loc[0, "DRR_RMSE"] = self.DRR_RMSE(h_dr[0,:,m],h_dr[1,:,m], d_dr[0,:,m],d_dr[1,:,m])
                results_dataframe_m.loc[0, "EDC_RMSE"] = self.EDC_RMSE(h_arr[:,m], d_arr[:,m])
                results_dataframe_m.loc[0, "C50_E"] = self.C50_E(h_arr[:,m], d_arr[:,m])
                results_dataframe_m.loc[0, "C50_GFB_RMSE"] = self.C50_GFB_RMSE(h_arr[:,m], d_arr[:,m])
                append_dataframe_to_csv(self.results_data_path, results_dataframe_m)

            # Binaural measures
            for m in range(2):
                results_dataframe_m = results_dataframe
                results_dataframe_m.loc[0, "Receiver_type"] = "Binaural"
                results_dataframe_m.loc[0, "Receiver"] = m
                results_dataframe_m.loc[0, "Spectral_RMSE"] = self.Spectral_RMSE(h_bin[:,m], d_bin[:,m])
                if m == 0:
                    results_dataframe_m.loc[0, "IACC_RMSE"] = self.IACC_RMSE(h_bin, d_bin)
                else:
                    results_dataframe_m.loc[0, "IACC_RMSE"] = float('NaN')
                results_dataframe_m.loc[0, "DRR_RMSE"] = self.DRR_RMSE(h_dr_bin[0,:,m],h_dr_bin[1,:,m], d_dr_bin[0,:,m],d_dr_bin[1,:,m])
                results_dataframe_m.loc[0, "EDC_RMSE"] = self.EDC_RMSE(h_bin[:,m], d_bin[:,m])
                results_dataframe_m.loc[0, "C50_E"] = self.C50_E(h_bin[:,m], d_bin[:,m])
                results_dataframe_m.loc[0, "C50_GFB_RMSE"] = self.C50_GFB_RMSE(h_bin[:,m], d_bin[:,m])
                append_dataframe_to_csv(self.results_data_path, results_dataframe_m)