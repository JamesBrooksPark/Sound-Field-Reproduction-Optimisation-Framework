

from src.filter_designer.data_helper import direct_reverberant_simulation, data_slice


def Run(Fs):
    # Read metadata
    meta_data = 01_Data/01_OptimisationData/metadata.csv
    # Loop through metadata
    
    # Read metadata for condition
    
    
    # Read filter data
    
    
    # Read DR rir data
    Filter_class = data_slice(num_sources, M_calculation, M_evaluation, 'square', 'square', Filter_class)
    
    # Run DR sims
    direct_reverberant_simulation(Filter_class.playback_array_evaluation_direct, Filter_class.playback_array_evaluation_reverberant, Filter_class.playback_binaural_direct, Filter_class.playback_binaural_reverberant, w, Filter_class.target_array_evaluation_direct, Filter_class.target_array_evaluation_reverberant, Filter_class.target_binaural_direct, Filter_class.target_binaural_reverberant, data_folder, id, Fs,perceived_rev)

if __name__ == "__main__":
    Fs = 44100
    data_loc = '/Users/jamesbrookspark/Documents/04_Repositories/RoomAugmentation-Analysis/01_Data'
    Run(Fs, data_loc)