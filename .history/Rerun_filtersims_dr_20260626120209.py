import os; import pandas as pd; import soundfile as sf; import numpy as np; import scipy as sp

from src.filter_designer.data_helper import direct_reverberant_simulation, data_slice


def Run(Fs, data_loc):
    # Read metadata
    meta_data_loc = os.path.join(data_loc,'01_OptimisationData/metadata.csv')
    meta_data = pd.read_csv(meta_data_loc)
    
    # Loop through metadata
    for index, row in meta_data.iterrows():
        # Read filter data
        id = row['id']
        filter_path = os.path.join(data_loc,f'01_OptimisationData/filters/filter_{id}.wav')
        w,Fs = sf.read(filter_path)
        
        # Read DR rir data
        h_arr_d, h_arr_r, h_bin_d, h_bin_r, d_arr_d, d_arr_r, d_bin_d, d_bin_r = dr_evaluation_import(data_loc, id)

        # # Run DR sims
        if row['ommit_primary'] and row['delay_supporting']:
            perceived_rev = True
        else:
            perceived_rev = False
    
        direct_reverberant_simulation(h_arr_d, h_arr_r, h_bin_d, h_bin_r, w, d_arr_d, d_arr_r, d_bin_d, d_bin_r, data_folder, id, Fs,perceived_rev)

def dr_evaluation_import(data_loc, id):
    
    bin_dir, bin_rev
    
    return h_arr_d, h_arr_r, h_bin_d, h_bin_r, d_arr_d, d_arr_r, d_bin_d, d_bin_r

def DR_import_arr(Room, M, array, data_loc):
    h_dir = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Array/RazrSim_Room{Room}_{array}_L2_M_{M}_Direct.mat')
    Room_dir = h_dir['room_direct']
    h_rev = sp.io.loadmat(f'/dss/work/nues6312/DATA/RoomAugmentation_10/Array/RazrSim_Room{Room}_{array}_L2_M_{M}_Reverberant.mat')
    Room_rev = h_rev['room_rev']
    Room = np.zeros([np.shape(Room_dir)[0], np.shape(Room_dir)[1], np.shape(Room_dir)[2], 2])
    Room[:,:,:,0] = Room_dir
    Room[:,:,:,1] = Room_rev
    
    return Room

def DR_import_bin(Room, data_loc):
    h_bin_dir = sp.io.loadmat(os.path.join(data_loc,f'01_Data/01_OptimisationData/DATA/RoomAugmentation_10/Binaural/RazrSim_Room{Room}_L2_Binaural_dir.mat'))
    Room_dir = h_bin_dir['room_direct']
    h_bin_rev = sp.io.loadmat(os.path.join(data_loc,f'01_Data/01_OptimisationData/DATA/RoomAugmentation_10/Binaural/RazrSim_Room{Room}_L2_Binaural_rev.mat'))
    Room_rev = h_bin_rev['room_rev']
    
    return Room_dir, Room_rev

if __name__ == "__main__":
    Fs = 44100
    data_loc = '/Users/jamesbrookspark/Documents/04_Repositories/RoomAugmentation-Analysis/01_Data'
    Run(Fs, data_loc)