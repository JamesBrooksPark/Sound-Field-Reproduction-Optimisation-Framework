import os; import pandas as pd; import soundfile as sf; import numpy as np

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
                
        # # Run DR sims
        delay_supporting,ommit_primary
        if row['ommit_primary'] and Filter_class.delay_supporting:
            perceived_rev = True
        else:
            perceived_rev = False
    
        direct_reverberant_simulation(h_arr_d, h_arr_r, h_bin_d, h_bin_r, w, d_arr_d, d_arr_r, d_bin_d, d_bin_r, data_folder, id, Fs,perceived_rev)

def dr_evaluation_import(data_loc, id):
    
    
    return h_arr_d, h_arr_r, h_bin_d, h_bin_r, d_arr_d, d_arr_r, d_bin_d, d_bin_r
if __name__ == "__main__":
    Fs = 44100
    data_loc = '/Users/jamesbrookspark/Documents/04_Repositories/RoomAugmentation-Analysis/01_Data'
    Run(Fs, data_loc)