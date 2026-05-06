import matplotlib.pyplot as plt; import numpy as np; import soundfile as sf; import os; import re;  from pathlib import Path

from src.ComplexSmoothing import Smoothing

class Plotting():
    def __init__(self,Fs):
        self.Fs = Fs
        self.dB_LowLim = -20
        self.dB_UpLim = 10

        self.time_lim = 1

    def Frequency(self, ir, axes=None, label=None, colour = 'blue'):
        """
        Plot the frequency response of the input signal `ir`.

        Parameters
        ----------
        ir : array_like
            Input signal.
        axes : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure/axes will be created.
        label : str, optional
            Label for the plot (useful for legends if plotting multiple curves).
        """
        NFFT = 2**13

        Sm = Smoothing(2**13)
        ir = Smoothing.complexSmoothing(Sm, ir, "spectrum", 1, 1 / 6)
       
        IR = np.abs(np.fft.fft(ir, NFFT))
        xlim = round(NFFT / 2)

        x = self.Fs / 2 * np.linspace(0, 1, xlim)
        y = 20 * np.log10(IR[0:xlim])

        created_new_fig = False
        if axes is None:
            fig, axes = plt.subplots()
            created_new_fig = True

        
        axes.plot(x, y, label=label, color=colour)

        axes.set_xscale("log")
        axes.set_xlabel("Frequency (Hz)")
        axes.set_ylabel("Magnitude (dB)")
        # axes.set_title("Frequency Plot")

        axes.set_xlim(left=20, right=20000)
        axes.set_ylim(bottom=self.dB_LowLim, top=self.dB_UpLim)

        axes.grid()

        if label:
            axes.legend()

        # Show only if the function created the figure itself
        if created_new_fig:
            plt.show()

        return axes
    
    def Time(self, ir, axes=None, label=None):
        """
        Plot the time response of the input signal `ir`.

        Parameters
        ----------
        ir : array_like
            Input signal.
        axes : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure/axes will be created.
        label : str, optional
            Label for the plot (useful for legends if plotting multiple curves).
        """
    
        y = ir
        
        x = np.arange(len(ir)) / self.Fs
        
        created_new_fig = False
        if axes is None:
            fig, axes = plt.subplots()
            created_new_fig = True
        axes.plot(x, y, label=label)

        axes.set_xlabel("time (s)")
        axes.set_ylabel("Time Plot")

        axes.set_ylim(bottom=-1, top=1)

        axes.grid()

        if label:
            axes.legend()

        # Show only if the function created the figure itself
        if created_new_fig:
            plt.show()

        return axes
    
def RIR_Freq_plots(data_folder, filter_id, filename):

    target, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_d_eval.wav")
    h_opt, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_h_opt_eval.wav")
    h, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_h_eval.wav")

    if h.ndim > 1:
        M = np.shape(h)[1]
    else:
        M = 1

    fig, axes = plt.subplots(M, 1, sharex=True, figsize=(8, 2 * M))
    P = Plotting(Fs)
    if M == 1:
        P.Frequency(target,axes,'target', colour='black')
        P.Frequency(h,axes,'h', colour= 'red')
        P.Frequency(h_opt,axes,'h_opt', colour= 'blue')

    else:
        for m in range(M):
            ax = axes[m]
            P.Frequency(target[:,m],ax,'target', colour='black')
            P.Frequency(h[:,m],ax,'h', colour= 'red')
            P.Frequency(h_opt[:,m],ax,'h_opt', colour= 'blue')
            
    plt.tight_layout()


    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(filename)
    plt.close()

def RIR_Time_plots(data_folder, filter_id, filename):
    target, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_d_eval.wav")
    h_opt, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_h_opt_eval.wav")
    h, Fs = sf.read(f"{data_folder}rirs/rirs_{filter_id}_h_eval.wav")

    if h.ndim > 1:
        M = np.shape(h)[1]
    else:
        M = 1

    fig, axes = plt.subplots(M * 3, 1, sharex=True, figsize=(8, 2 * M * 3))
    
    P = Plotting(Fs)
    P.time_lim = np.shape(h_opt)[0] / Fs
    if M == 1:
        P.Time(target, axes[0],'target')
        P.Time(h, axes[1],'h')
        P.Time(h_opt, axes[2],'h_opt')

    else:
        for m in range(M):
            P.Time(target[:,m],axes[int(m * 3)],'target')
            P.Time(h[:,m],axes[int((m * 3) + 1)],'h')
            P.Time(h_opt[:,m],axes[int((m * 3) + 2)],'h_opt')
            
    plt.tight_layout()

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(filename)
    plt.close()

def BRIR_Time_plots(data_folder, filter_id, filename):
    target, Fs = sf.read(f"{data_folder}brirs/brirs_{filter_id}_d_b.wav")
    h_opt, Fs = sf.read(f"{data_folder}brirs/brirs_{filter_id}_h_opt_b.wav")
    h, Fs = sf.read(f"{data_folder}brirs/brirs_{filter_id}_h_b.wav")

    M = 2

    fig, axes = plt.subplots(M * 3, 1, sharex=True, figsize=(8, 2 * M * 3))
    
    P = Plotting(Fs)
    P.time_lim = np.shape(h_opt)[0] / Fs
    if M == 1:
        P.Time(target, axes[0],'target')
        P.Time(h, axes[1],'h')
        P.Time(h_opt, axes[2],'h_opt')

    else:
        for m in range(M):
            P.Time(target[:,m],axes[int(m * 3)],'target')
            P.Time(h[:,m],axes[int((m * 3) + 1)],'h')
            P.Time(h_opt[:,m],axes[int((m * 3) + 2)],'h_opt')
            
    plt.tight_layout()

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(filename)
    plt.close()

def Filter_Time_plots(data_folder, filter_id, filename):
    filter, Fs = sf.read(f"{data_folder}filters/filter_{filter_id}.wav")
        
    L = np.shape(filter)[1] if filter.ndim > 1 else 1

    fig, axes = plt.subplots(L, 1, sharex=True, figsize=(8, 2 * L))
    
    P = Plotting(Fs)
    P.time_lim = np.shape(filter)[0] / Fs
    if L == 1:
        P.Time(filter,axes,'target')

    else:
        for l in range(L):
            ax = axes[l]
            P.Time(filter[:,l],ax,'target')
            
    plt.tight_layout()

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(filename)
    plt.close()
    
def Source_Frequency_plots(data_folder, filter_id, filename):
    source_folder = f"{data_folder}/sources/"
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
    L = source_N_finder(source_folder, filter_id)
    P = Plotting(44100)
    for l in range(L):
        
        h, Fs = sf.read(f"{source_folder}id_{filter_id}_source_{l}_h.wav")
        h_opt, Fs = sf.read(f"{source_folder}id_{filter_id}_source_{l}_h_opt.wav")
        
        M = np.shape(h)[1] if h.ndim > 1 else 1
            
        fig, axes = plt.subplots(M * 2, 1, sharex=True, figsize=(8, 2 * M * 2))
        
        if M == 1:
            P.Frequency(h,axes[0],'h')
            P.Frequency(h_opt,axes[1],'h_opt')
        else:
            for m in range(M):
                P.Frequency(h[:,m],axes[(m*2)+0],'h')
                P.Frequency(h_opt[:,m],axes[(m*2)+1],'h_opt')
            
        filename_mod = f"{filename[:-4]}_source_{l}.pdf"
        plt.tight_layout()

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(filename_mod)
        plt.close()
    
def source_N_finder(data_folder, target_id):
    """
    Counts the number of unique sources for a given id in the folder filenames.
    """
    id_pattern = re.compile(r"id_(\d+)_")
    source_pattern = re.compile(r"source_(\d+)_")
    
    sources = set()  # to store unique source numbers

    for filename in os.listdir(data_folder):
        id_match = id_pattern.search(filename)
        if id_match and int(id_match.group(1)) == target_id:
            source_match = source_pattern.search(filename)
            if source_match:
                sources.add(int(source_match.group(1)))

    return len(sources)