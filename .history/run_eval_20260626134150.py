import os; import shutil

from src.numerical_evaluation.numerical_measures import NumericalMeasures

from src.plotting.plotting_functions import *

def run_plotting():
    print("Plotting")
    data_folder = '/dss/work/nues6312/SoFROF/data/'
    filter_folder = data_folder + 'filters/'
    figures_folder = data_folder + 'figures/'
    
    files = os.scandir(filter_folder)
    N = 0
    for entry in files:
        if entry.name.endswith(".wav"):
            N += 1
    
    if os.path.exists(figures_folder) and os.path.isdir(figures_folder):
        shutil.rmtree(figures_folder)
        
    for ii in range(N):
        frequency_filename = figures_folder + f"frequency_plots/{ii}_frequency.pdf"
        RIR_Freq_plots(data_folder, ii, frequency_filename)

        rir_filename = figures_folder + f"rirs/{ii}_rir.pdf"
        RIR_Time_plots(data_folder, ii, rir_filename)

        brir_filename = figures_folder + f"/brirs/{ii}_brir.pdf"
        BRIR_Time_plots(data_folder, ii, brir_filename)

        filter_filename = figures_folder +  f"/filters/{ii}_filter.pdf"
        Filter_Time_plots(data_folder, ii, filter_filename)
        
        # filter_filename = figures_folder + f"/sources/{ii}.pdf"
        # Source_Frequency_plots(data_folder, ii, filter_filename)
        
def run_NE(Fs):
    print("NE")
    NM = NumericalMeasures(Fs)
    NM.meta_data_path = '/dss/work/nues6312/SoFROF/data/metadata.csv'
    NM.results_data_path = '/dss/work/nues6312/SoFROF/data/NEresults.csv'
    NM.data_path = '/dss/work/nues6312/SoFROF/data'
    
    NM.meta_data_path = '/Users/jamesbrookspark/Documents/04_Repositories/RoomAugmentation-Analysis/01_Data/01_OptimisationData/metadata.csv'
    NM.results_data_path = '/dss/work/nues6312/SoFROF/data/NEresults.csv'
    NM.data_path = '/dss/work/nues6312/SoFROF/data'
    
    NM.run_measures()


if __name__ == "__main__":
    Fs = 44100
    run_plotting()
    run_NE(Fs)