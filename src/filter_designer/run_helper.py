def define_conditions():
    # Set constant parameters:
    calculation_M = 9
    evaluation_M = 25
    
    playback_index = 5
    
    # Define variable parameters
    target_rooms = [6]
    num_sources_vec = [1,3]
    optimisation_criteria = ["pressure","amplitude","mixed"]
    delay_supporting_bools = [True, False]
    ommit_primary_bools = [True, False]
    
    playback_sofa_dir = '/dss/work/nues6312/DATA/RoomAugmentation_10/Binaural/Sofa/Room5'
    
    conditions = []
    idx = 0
    for target_index in target_rooms:
        for num_sources in num_sources_vec:
            for delay_supporting_bool in delay_supporting_bools:
                for ommit_primary_bool in ommit_primary_bools:
                    if num_sources == 1 and (delay_supporting_bool or ommit_primary_bool):
                        continue
                    else:
                        for optimisation_criterion in optimisation_criteria:
                            condition = {
                                "index":idx,
                                "playback_index":playback_index,
                                "target_index" :target_index,
                                "num_sources": num_sources,
                                "num_receivers_calc":calculation_M,
                                "num_receivers_eval":evaluation_M,
                                "optimisation_criteria": optimisation_criterion,
                                "delay_supporting": delay_supporting_bool,
                                "ommit_primary": ommit_primary_bool,
                                "playback_sofa_dir":playback_sofa_dir
                            }
                            conditions.append(condition)
                            idx+=1
        # RoomInRoom
        room_in_room_condition = {
            "index":idx,
            "playback_index":playback_index,
            "target_index" :target_index,
            "num_sources": 1,
            "num_receivers_calc":calculation_M,
            "num_receivers_eval":evaluation_M,
            "optimisation_criteria": "RoomInRoom",
            "delay_supporting": False,
            "ommit_primary": False,
            "playback_sofa_dir":playback_sofa_dir
        }
        conditions.append(room_in_room_condition)
        idx+=1
        
    # Playback sim
    uncomp_condition = {
            "index":idx,
            "playback_index":playback_index,
            "target_index" :target_index,
            "num_sources": 1,
            "num_receivers_calc":calculation_M,
            "num_receivers_eval":evaluation_M,
            "optimisation_criteria": "Uncomp",
            "delay_supporting": False,
            "ommit_primary": False,
            "playback_sofa_dir":playback_sofa_dir
        }
    conditions.append(uncomp_condition)
    
    return conditions, idx
   