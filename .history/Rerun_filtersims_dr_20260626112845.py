import os; import pandas as pd; import soundfile as sf

from src.filter_designer.data_helper import direct_reverberant_simulation, data_slice


def Run(Fs, data_loc):
    # Read metadata
    meta_data_loc = os.path.join(data_loc,'01_OptimisationData/metadata.csv')
    meta_data = pd.read_csv(meta_data_loc)
    
    # Loop through metadata
    for index, row in meta_data.iterrows():
        # Read filter data
        id = row['id']
        filter_path = os.path.join(data_loc,'')
        # sf.read()
        
        # Read DR rir data
        # Filter_class = data_slice(num_sources, M_calculation, M_evaluation, 'square', 'square', Filter_class)
        
        # # Run DR sims
        # direct_reverberant_simulation(Filter_class.playback_array_evaluation_direct, Filter_class.playback_array_evaluation_reverberant, Filter_class.playback_binaural_direct, Filter_class.playback_binaural_reverberant, w, Filter_class.target_array_evaluation_direct, Filter_class.target_array_evaluation_reverberant, Filter_class.target_binaural_direct, Filter_class.target_binaural_reverberant, data_folder, id, Fs,perceived_rev)

if __name__ == "__main__":
    Fs = 44100
    data_loc = '/Users/jamesbrookspark/Documents/04_Repositories/RoomAugmentation-Analysis/01_Data'
    Run(Fs, data_loc)