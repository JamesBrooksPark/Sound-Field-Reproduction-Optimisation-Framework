import sys; import json
from pathlib import Path

from src.filter_designer.run_helper import define_conditions
from src.filter_designer.filter_calculate import run_condition

if __name__ == "__main__":
    op = 'cluster'
    
    # All file paths:
    conditions_filepath = '/dss/work/nues6312/SoFROF/data/conditions.json'
    metadata_filename = '/dss/work/nues6312/SoFROF/data/metadata.csv'
    data_folder = '/dss/work/nues6312/SoFROF/data/'
    
    
    conds, N = define_conditions()
    
    if op == 'cluster':
        idx = int(sys.argv[1]) 
        if idx == 0:
            Path(conditions_filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(conditions_filepath, "w", encoding="utf-8") as f:
                json.dump(conds, f, indent=4)
                
        run_condition(metadata_filename, data_folder, conds[idx])
        
    elif op == 'single':
        with open(conditions_filepath, "w", encoding="utf-8") as f:
            json.dump(conds, f, indent=4)
        for n in range(N):
            run_condition(metadata_filename, data_folder, conds[n])