import numpy as np; import time

from src.filter_designer.filter_designer import FilterDesigner
from src.filter_designer.data_helper import data_slice, create_data_frame, append_dataframe_to_csv, simulations, sofa_simulation, save_audio_file, save_irs, direct_reverberant_simulation

# ++ To-Do ++
# - FIR shaper normalisation

def run_condition(metadata_filename, data_folder, condition):
    print("Running...")
    
    id = condition["index"]
    
    Fs = 44100
    num_sources = condition["num_sources"]
    M_calculation = condition["num_receivers_calc"]
    M_evaluation = condition["num_receivers_eval"]
        
    
    # define parameters from condition
    Filter_class = FilterDesigner(Fs)
    Filter_class.playback_index = condition["playback_index"]
    Filter_class.target_index = condition["target_index"]
    Filter_class.delay_supporting = condition["delay_supporting"]
    Filter_class.ommit_primary = condition["ommit_primary"]
    Filter_class.optimisation_criteria = condition["optimisation_criteria"]

    playback_sofa_dir = condition["playback_sofa_dir"]
    
    # Import data
    Filter_class = data_slice(num_sources, M_calculation, M_evaluation, 'square', 'square', Filter_class)
    
    # Start timer
    start = time.time()
    
    # Filter designer
    w, meta_data = Filter_class.filter_designer()
    
    # Stop timer
    end = time.time()
    t = np.round(end - start,2)

    # Complete meta data 
    meta_data['id'] = id
    meta_data['target_room'] = f'Room_{Filter_class.target_index}'
    meta_data['playback_room'] = f'Room_{Filter_class.playback_index}'
    
    # Save meta data
    meta_data_df = create_data_frame(meta_data, t)
    append_dataframe_to_csv(metadata_filename, meta_data_df)
    
    # Simulations
    h_arr_opt = simulations(Filter_class.playback_array_calculation,w)
    h_arr_opt_rand = simulations(Filter_class.playback_array_evaluation,w)
    h_bin_opt = simulations(Filter_class.playback_binaural,w)
    sofa_save_path = data_folder + "/sofa/"
    sofa_simulation(playback_sofa_dir, sofa_save_path, w, str(id))
    
    
    # Save Filter and rirs
    filename_filter = f"{data_folder}/filters/filter_{id}.wav"
    save_audio_file(w, filename_filter, Fs)
    
    save_irs(Filter_class.playback_array_calculation, h_arr_opt, Filter_class.target_array_calculation, data_folder, id, Fs, 'array')
    save_irs(Filter_class.playback_array_evaluation, h_arr_opt_rand, Filter_class.target_array_evaluation, data_folder, id, Fs, 'array_eval')
    save_irs(Filter_class.playback_binaural,h_bin_opt , Filter_class.target_binaural, data_folder, id, Fs, 'binaural')   
    
    if Filter_class.ommit_primary and Filter_class.delay_supporting:
        perceived_rev = True
    else:
        
    
    direct_reverberant_simulation(Filter_class.playback_array_evaluation_direct, Filter_class.playback_array_evaluation_reverberant, Filter_class.playback_binaural_direct, Filter_class.playback_binaural_reverberant, w, Filter_class.target_array_evaluation_direct, Filter_class.target_array_evaluation_reverberant, Filter_class.target_binaural_direct, Filter_class.target_binaural_reverberant, data_folder, id, Fs, Filter_class.ommit_primary)