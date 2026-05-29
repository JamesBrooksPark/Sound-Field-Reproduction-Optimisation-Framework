import numpy as np; import scipy as sp; import os; import pandas as pd; from scipy.signal import fftconvolve; import sofar as sofa; import soundfile as sf; from pathlib import Path

# -----------------------------
# ===== Importing of data =====
# -----------------------------
def data_slice(L, M_calc, M_eval, array_type_calc, array_type_eval, Class):
    # Naming of arrays:
    # Playback / Target
    # Array / Binaural
    
    # Optional 
    # Calculation / Evaluation
    # Direct / Reverberant
    
    playback_index = Class.playback_index
    target_index = Class.target_index
    
    # Playback Array Calculation
    array, Fs = RIR_import(playback_index, M_calc, array_type_calc)
    Class.playback_array_calculation = array[:,:L,:]
    
    # Playback Array Evaluation
    array, Fs = RIR_import(playback_index, M_eval, array_type_eval)
    Class.playback_array_evaluation = array[:,:L,:]
    
    # Playback Binaural
    array, Fs = BRIR_import(playback_index)
    Class.playback_binaural = array[:,:L,:]
    
    # Playback Array Calculation DR
    array = DR_import_arr(playback_index, M_calc, array_type_calc)
    array = array[:,:L,:,:]
    Class.playback_array_calculation_direct = array[:,:,:,0]
    Class.playback_array_calculation_reverberant = array[:,:,:,1]
    
    # Playback Array Evaluation DR
    array = DR_import_arr(playback_index, M_eval, array_type_eval)
    array = array[:,:L,:,:]
    Class.playback_array_evaluation_direct = array[:,:,:,0]
    Class.playback_array_evaluation_reverberant = array[:,:,:,1]
    
    # Playback Binaural DR
    array = DR_import_bin(playback_index)
    array = array[:,:L,:,:]
    Class.playback_binaural_direct = array[:,:,:,0]
    Class.playback_binaural_reverberant = array[:,:,:,1]
    
    # Target Array Calculation
    array, Fs = RIR_import(target_index, M_calc, array_type_calc)
    Class.target_array_calculation = np.squeeze(array[:,:1,:])
    
    # Target Array Evaluation
    array, Fs = RIR_import(target_index, M_eval, array_type_eval)
    Class.target_array_evaluation = np.squeeze(array[:,:1,:])
    
    # Target Binaural
    array, Fs = BRIR_import(target_index)
    Class.target_binaural = np.squeeze(array[:,:1,:])
    
    # Target Array Calculation DR
    array = DR_import_arr(target_index, M_calc, array_type_calc)
    array = array[:,:1,:,:]
    Class.target_array_calculation_direct = np.squeeze(array[:,:1,:,0])
    Class.target_array_calculation_reverberant = np.squeeze(array[:,:1,:,1])
    
    # Target Array Evaluation DR
    array = DR_import_arr(target_index, M_eval, array_type_eval)
    array = array[:,:1,:,:]
    Class.target_array_evaluation_direct = np.squeeze(array[:,:1,:,0])
    Class.target_array_evaluation_reverberant = np.squeeze(array[:,:1,:,1])
    
    # Target Binaural DR
    array = DR_import_bin(target_index)
    array = array[:,:L,:,:]
    Class.target_binaural_direct = np.squeeze(array[:,:1,:,0])
    Class.target_binaural_reverberant = np.squeeze(array[:,:1,:,1])

    Class.Fs = Fs
    Class.array_type = array_type_calc
    
    return Class

def RIR_import(Room, M, array):
    h_arr = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Array/RazrSim_Room{Room}_{array}_L2_M_{M}.mat')
    Room_array = h_arr['room_rir']
    Fs = 44100
    
    return Room_array, Fs

def BRIR_import(Room):
    h_bin = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Binaural/RazrSim_Room{Room}_L2_Binaural.mat')
    Room_bin = h_bin['room_rir']
    Fs = 44100
    
    return Room_bin, Fs

def DR_import_arr(Room, M, array):
    h_dir = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Array/RazrSim_Room{Room}_{array}_L2_M_{M}_Direct.mat')
    Room_dir = h_dir['room_direct']
    h_rev = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Array/RazrSim_Room{Room}_{array}_L2_M_{M}_Reverberant.mat')
    Room_rev = h_rev['room_rev']
    Room = np.zeros([np.shape(Room_dir)[0], np.shape(Room_dir)[1], np.shape(Room_dir)[2], 2])
    Room[:,:,:,0] = Room_dir
    Room[:,:,:,1] = Room_rev
    
    return Room

def DR_import_bin(Room):
    h_bin_dir = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Binaural/RazrSim_Room{Room}_L2_Binaural_dir.mat')
    Room_dir = h_bin_dir['room_direct']
    h_bin_rev = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Binaural/RazrSim_Room{Room}_L2_Binaural_rev.mat')
    Room_rev = h_bin_rev['room_rev']
    Room = np.zeros([np.shape(Room_dir)[0], np.shape(Room_dir)[1], 2, 2])
    Room[:,:,:,0] = Room_dir
    Room[:,:,:,1] = Room_rev
    
    return Room


# --------------------------------------
# ===== Meta data helper functions =====
# --------------------------------------

def create_data_frame(meta_data, t):
    df = pd.DataFrame({
        "id": meta_data['id'],
        "N_sources": meta_data['N_sources'],
        "N_receivers": meta_data['N_receivers'],
        "Receiver Geom": meta_data['Receiver_Geom'],
        "method": meta_data['method'],
        "delay_supporting": meta_data['delay_supporting'],
        "ommit_primary": meta_data['ommit_primary'],
        "filter_length": meta_data['filter_length'],
        "NFFT": meta_data['NFFT'],
        "Modeling delay": meta_data['modelling_delay'],
        "Regularisation weight": meta_data['regularisation_weight'],
        "filter shaping weight": meta_data['FIR_shaping_weight'],
        "gamma_value_lower": meta_data['gamma_value_lower'],
        "gamma_value_higher": meta_data['gamma_value_higher'],
        "gamma_crossover": meta_data['gamma_crossover'],
        "target_room": meta_data['target_room'],
        "playback_room": meta_data['playback_room'],
        "run time ": t
    }, index=[0])
    return df

def append_dataframe_to_csv(filename, df_new):
    # Check if file exists and is non-empty
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df_existing = pd.read_csv(filename)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new.copy()  # Just use the new DataFrame
    
    # Save back to CSV
    df_updated.to_csv(filename, index=False)
    
    
# --------------------------------------
# ===== Simulation and Saving functions =====
# --------------------------------------
def simulations(h, w):
    # Ensure h is 3D (J, L, M) and w is 2D (I, L)
    h = np.atleast_3d(h)
    w = np.atleast_2d(w).T if np.ndim(w) == 1 else w
    
    J, L, M = h.shape
    I = w.shape[0]
    
    # Pre-allocate the output array
    h_opt = np.zeros((J + I - 1, M))
    
    # Iterate only over the necessary dimensions
    for m in range(M):
        for l in range(L):
            h_opt[:, m] += fftconvolve(h[:, l, m], w[:, l], 'full')
        
    return h_opt

def sofa_simulation(dir, save_dir, w, filename):   
    # if save directory does not exist, create it
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    if w.ndim > 1:
        L = np.shape(w)[1]
    else:
        L = 1
                
    # n = 0
    # files = os.scandir(dir)
    # for entry in files:
    #     if entry.name.endswith(".sofa"):
    #         if n < L:
    #             sofa_read = sofa.read_sofa(entry.path)
    #             if n == 0:
    #                     if L > 1:
    #                         data = filter_sofa_data(sofa_read.Data_IR ,w[:,n])
    #                     else:
    #                         data = filter_sofa_data(sofa_read.Data_IR ,w)
    #             else:
    #                 data += filter_sofa_data(sofa_read.Data_IR ,w[:,n])
    #             n += 1
    for l in range(L):
        sofa_path = dir + f'/RazrSim__L{l+1}__3_Binaural.sofa'
        sofa_read = sofa.read_sofa(sofa_path)
        if l == 0:
            if L > 1:
                data = filter_sofa_data(sofa_read.Data_IR ,w[:,l])
            else:
                data = filter_sofa_data(sofa_read.Data_IR ,w)
        else:
            data += filter_sofa_data(sofa_read.Data_IR ,w[:,l])
            
            
    sofa_save = sofa_read
    sofa_save.Data_IR = data
    save_path = save_dir + filename + '.sofa'
    sofa.write_sofa(save_path, sofa_save)
    
def filter_sofa_data(data,w):
    J = np.shape(data)[2]
    N = np.shape(data)[0]
    I = len(w)
    data_out = np.zeros([N,2,J+I-1])
    for n in range(N):
        data_out[n,0,:] = fftconvolve(data[n,0,:],w,'full')
        data_out[n,1,:] = fftconvolve(data[n,1,:],w,'full')
        
    return data_out
    
def save_audio_file(file, filename, Fs):
    # Extract the directory from the filename
    folder = os.path.dirname(filename)
    
    # Check if the folder exists, create if not
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save the audio file
    sf.write(filename, file, Fs)
    
def save_irs(h,h_opt, d, data_folder, id, Fs, type):
    h = np.atleast_3d(h)
    J,L,M = np.shape(h)
    h = h[:,0,:]
            
    if M == 1:
        Len = max(len(h), len(h_opt), len(d))
        data_out = np.zeros([Len, 3])
        data_out[:,0] = zero_padding(h,Len)
        data_out[:,1] = zero_padding(h_opt,Len)
        data_out[:,2] = zero_padding(d,Len)
        
    if M > 1:
        Len = max(len(h[:,0]), len(h_opt[:,0]), len(d[:,0]))
        data_out = np.zeros([Len,3, M])
        for m in range(M):
            data_out[:,0, m] = zero_padding(h[:,m],Len)
            data_out[:,1, m] = zero_padding(h_opt[:,m],Len)
            data_out[:,2, m] = zero_padding(d[:,m],Len)
            
    # Save files
    if type == 'binaural':
        save_audio_file(data_out[:,0,:], f"{data_folder}/brirs/brirs_{id}_h_b.wav", Fs)
        save_audio_file(data_out[:,1,:], f"{data_folder}/brirs/brirs_{id}_h_opt_b.wav", Fs)
        save_audio_file(data_out[:,2,:], f"{data_folder}/brirs/brirs_{id}_d_b.wav", Fs)
    elif type == 'array':
        if M == 1:
            save_audio_file(data_out[:,0], f"{data_folder}/rirs/rirs_{id}_h.wav", Fs)
            save_audio_file(data_out[:,1], f"{data_folder}/rirs/rirs_{id}_h_opt.wav", Fs)
            save_audio_file(data_out[:,2], f"{data_folder}/rirs/rirs_{id}_d.wav", Fs)
        else:
            save_audio_file(data_out[:,0,:], f"{data_folder}/rirs/rirs_{id}_h.wav", Fs)
            save_audio_file(data_out[:,1,:], f"{data_folder}/rirs/rirs_{id}_h_opt.wav", Fs)
            save_audio_file(data_out[:,2,:], f"{data_folder}/rirs/rirs_{id}_d.wav", Fs)
    elif type == 'array_eval':
        if M == 1:
            print("No random recivers for binaural")
        else:
            save_audio_file(data_out[:,0,:], f"{data_folder}/rirs/rirs_{id}_h_eval.wav", Fs)
            save_audio_file(data_out[:,1,:], f"{data_folder}/rirs/rirs_{id}_h_opt_eval.wav", Fs)
            save_audio_file(data_out[:,2,:], f"{data_folder}/rirs/rirs_{id}_d_eval.wav", Fs)
    else:
        ValueError("Unknown type for saving RIRs")
        
def zero_padding(x,L):
    return np.pad(x, (0, L - len(x)), mode='constant')
        
def direct_reverberant_simulation(h_arr_dir, h_arr_rev, h_bin_dir, h_bin_rev, w, d_arr_dir, d_arr_rev, d_bin_dir, d_bin_rev, data_folder, id, Fs, support):
    w = np.atleast_2d(w) if w.ndim == 1 else w.T
    L = w.shape[0]


    for l in range(L):
        w_l = w[l:l+1,:].T
        
        if l == 0:
            arr_dir_conv = fftconvolve(h_arr_dir[:,l,:],w_l,'full', axes = 0) 
            arr_rev_conv = fftconvolve(h_arr_rev[:,l,:],w_l,'full', axes = 0)
            bin_dir_conv = fftconvolve(h_bin_dir[:,l,:],w_l,'full', axes = 0)
            bin_rev_conv = fftconvolve(h_bin_rev[:,l,:],w_l,'full', axes = 0)
        else:
            if support:
                arr_rev_conv += fftconvolve(h_arr_dir[:,l,:],w_l,'full', axes = 0) 
                bin_rev_conv += fftconvolve(h_bin_dir[:,l,:],w_l,'full', axes = 0)
            else:
                arr_dir_conv += fftconvolve(h_arr_dir[:,l,:],w_l,'full', axes = 0) 
                bin_dir_conv += fftconvolve(h_bin_dir[:,l,:],w_l,'full', axes = 0)
            
            arr_rev_conv += fftconvolve(h_arr_rev[:,l,:],w_l,'full', axes = 0)
            bin_rev_conv += fftconvolve(h_bin_rev[:,l,:],w_l,'full', axes = 0)
            
    save_audio_file(arr_dir_conv, f"{data_folder}/drr/h_opt_eval_{id}_dir.wav", Fs)
    save_audio_file(arr_rev_conv, f"{data_folder}/drr/h_opt_eval_{id}_rev.wav", Fs)
    save_audio_file(bin_dir_conv, f"{data_folder}/drr/h_opt_brir_{id}_dir.wav", Fs)
    save_audio_file(bin_rev_conv, f"{data_folder}/drr/h_opt_brir_{id}_rev.wav", Fs)
    

    save_audio_file(h_arr_dir[:,0,:], f"{data_folder}/drr/h_eval_{id}_dir.wav", Fs)
    save_audio_file(h_arr_rev[:,0,:], f"{data_folder}/drr/h_eval_{id}_rev.wav", Fs)
    save_audio_file(h_bin_dir[:,0,:], f"{data_folder}/drr/h_brir_{id}_dir.wav", Fs)
    save_audio_file(h_bin_rev[:,0,:], f"{data_folder}/drr/h_brir_{id}_rev.wav", Fs)
    
    save_audio_file(d_arr_dir, f"{data_folder}/drr/d_eval_{id}_dir.wav", Fs)
    save_audio_file(d_arr_rev, f"{data_folder}/drr/d_eval_{id}_rev.wav", Fs)
    save_audio_file(d_bin_dir, f"{data_folder}/drr/d_brir_{id}_dir.wav", Fs)
    save_audio_file(d_bin_rev, f"{data_folder}/drr/d_brir_{id}_rev.wav", Fs)