import numpy as np


# === Configuration file loader ===
# ---------------------------------
import importlib
class ConfigLoader:
    def __init__(self, config_file_name=None):
        config = self.load_config(config_file_name)  # config_file_name can be None
        self.set_config_variables(config)
    
    def set_config_variables(self, config):
        """Set configuration variables as instance attributes."""
        for key, value in config.items():
            setattr(self, key, value)

    def load_config(self, config_file_name=None):
        """Load base config and optionally update it with another config."""
        

        if config_file_name:
            config = self.import_config_module(config_file_name)
        else:
            config = self.import_config_module('global_config')

        return config

    def import_config_module(self, config_module_name):
        """Import a Python config module and return its non-special variables."""
        try:
            config_module = importlib.import_module(f'src.{config_module_name}')
            return {key: value for key, value in vars(config_module).items() if not key.startswith('__')}
        except ModuleNotFoundError:
            raise ValueError(f"Configuration file '{config_module_name}.py' not found.")
        
# === RIR matrix formating ===
# ----------------------------
def MatrixFormatting(h_d,h_p):
    # Finding start and end points of RIR's
    J_p,[p_s,p_e] = MatrixSlicer(h_p)
    J_d,[d_s,d_e]= MatrixSlicer(h_d)

    # Selecting largest RIR length
    e = np.max([p_e,d_e])

    # Selecting first start point across target and playback RIR's
    s = np.min([p_s,d_s])

    # Calculating rir size
    J = e-s
    
    N = h_p.ndim
    pad_shape = (J,) + h_p.shape[1:]
    h_p_pad = np.zeros(pad_shape, dtype=h_p.dtype)

    len_p = J_p + (p_s - s)

    slice_p = [slice(0, len_p)] + [slice(None)] * (N - 1)
    slice_p_src = [slice(s, p_e)] + [slice(None)] * (N - 1)

    h_p_pad[tuple(slice_p)] = h_p[tuple(slice_p_src)]

    N = h_d.ndim
    pad_shape = (J,) + h_d.shape[1:]
    h_d_pad = np.zeros(pad_shape, dtype=h_d.dtype)

    len_d = J_d + (d_s - s)

    slice_d = [slice(0, len_d)] + [slice(None)] * (N - 1)
    slice_d_src = [slice(s, d_e)] + [slice(None)] * (N - 1)

    h_d_pad[tuple(slice_d)] = h_d[tuple(slice_d_src)]
    
    h_d_pad = np.asarray(h_d_pad, dtype=np.float32)
    h_p_pad = np.asarray(h_p_pad, dtype=np.float32)
    
    return h_d_pad,h_p_pad,J

# === Matrix slicing ===
# ----------------------
def MatrixSlicer(h):
    h_shape = np.shape(h)
    if np.size(h_shape) < 2: # If matrix only has 1 dimension eg. single source, single receiver:
        start = PeakFind(h) - 15
        end = DecayFind(h)
        
        J = end-start
        return J,[start,end]
    elif np.size(h_shape) <3: # If matrix only has two dimensions eg. multiple sources but for a single receiver:
        deltas = np.zeros([h_shape[1], 2])
        for ii in range(h_shape[1]):
            deltas[ii, 0] = PeakFind(h[:, ii]) - 15
            deltas[ii, 1] = DecayFind(h[:, ii])
            
        start = int(np.min(deltas[:, 0]))
        end = int(np.min(deltas[:, 1]))
        J = end - start # Length of FIR representation of the RIRs
        return J,[start,end]
    else:
        deltas = np.zeros([h_shape[1], h_shape[2], 2])
        for ii in range(h_shape[1]):
            for jj in range(h_shape[2]):
                deltas[ii, jj, 0] = PeakFind(h[:, ii, jj]) - 15
                deltas[ii, jj, 1] = DecayFind(h[:, ii, jj])

        start = int(np.min(deltas[:, :, 0]))
        end = int(np.min(deltas[:, :, 1]))
        J = end - start # Length of FIR representation of the RIRs
        return J,[start,end]
def DecayFind(h):
        early_decay_curve = np.cumsum(h[::-1] ** 2)[::-1] # Calculate early decay curve of RIR
        early_decay_curve = early_decay_curve / np.max(early_decay_curve) # Normalise early decay curve to 1

        cut = np.where(early_decay_curve <= 0.000001)[0][0] # Find the point where the EDC is at 0.0001%
        return cut
def PeakFind(h):
        h_length = np.size(h)
        h_max = np.max(h)
        for ii in range(h_length):
            if h[ii] > h_max * 0.6:
                return ii
            
# === Precedence delay calculation ===
# -----------------------------
def DelayCalculation(Fs, delay_time, h_p, h_s):
        def peak(h):
            max = np.max(h)
            for ii in range(np.size(h)):
                if h[ii] > max * 0.6:
                    return ii
        
        if h_p.ndim > 1:
            M = np.shape(h_p)[1]
            h_p_peak_vec = np.zeros(M)
            h_s_peak_vec = np.zeros(M)
            for m in range(M):
                h_p_peak_vec[m] = peak(h_p[:,m])
                h_s_peak_vec[m] = peak(h_s[:,m])
            h_p_peak = np.sum(h_p_peak_vec)/M
            h_s_peak = np.sum(h_s_peak_vec)/M
            
            delay_samples = Fs * delay_time
            delay = int(delay_samples + (h_p_peak - h_s_peak))
            return delay
        else:
            h_p_peak = peak(h_p)
            h2_p_peak = peak(h_s)

            delay_samples = Fs * delay_time
            delay = int(delay_samples + (h_p_peak - h2_p_peak))
            return delay
        
def FIRShaper(L, Fl, delay, type, normalisation = False):
    if L == Fl:
        Fl = Fl-delay    
    z = np.zeros([3]) # Amplitudes
    z[0] = 10**16  # No energy
    z[1] = 1  # Max filter energy
    z[2] = 10**9  # taper energy
    
    type = 'exponential'
    if type == 'exponential':
        d = np.zeros([4]) # section sizes
        d[0] = int(delay)# Delay
        d[1] = int(100)  # Attack
        d[2] = int(Fl-d[1])  # Filter length (-1 for max filter sample)
        if L != Fl:
            d[3] = int(L - np.sum(d[:3]))
        else:
            d[3] = 0
        
        shaper_vec_check(L,Fl,d)
        
        y = np.ones([L])
        
        y[: int(np.sum(d[:1]))] *= z[0]
        y[int(np.sum(d[:1])) : int(np.sum(d[:2]))] = exponential_space(z[2], z[1], d[1])
        y[int(np.sum(d[:2]))] *= z[1]
        y[int(np.sum(d[:2]))+1 : int(np.sum(d[:3]))] = exponential_space(z[1], z[2], d[2]-1)
        if d[3]> 0:
            y[int(np.sum(d[:3])): ] = z[0]
            
    elif type == 'constant':
        d = np.zeros([5])
        d[0] = int(delay)
        d[1] = int(10) # Attack
        d[3] = int(200) # Decay
        d[2] = int(Fl-d[1]-d[3])
        if L != Fl:
            d[4] = int(L - np.sum(d[:4]))
        else:
            d[4] = 0
        
        shaper_vec_check(L,Fl,d)
        
        y = np.ones([L])
        y[: int(np.sum(d[:1]))] *= z[0]
        y[int(np.sum(d[:1])) : int(np.sum(d[:2]))] = exponential_space(z[0], z[1], d[1])
        y[int(np.sum(d[:2])) : int(np.sum(d[:3]))] *= z[1]
        y[int(np.sum(d[:3])) : int(np.sum(d[:4]))] = exponential_space(z[1], z[0], d[3])
        if d[4]> 0:
            y[int(np.sum(d[:4])): ] = z[0]
            
    # if normalisation:
    #     # Calculate energy in weighting with no delay
    #     d = np.zeros([5])
    #     d[0] = int(0)
    #     d[1] = int(10) # Attack
    #     d[3] = int(200) # Decay
    #     d[2] = int(Fl-d[1]-d[3])
    #     if L != Fl:
    #         d[4] = int(L - np.sum(d[:4]))
    #     else:
    #         d[4] = 0
        
    #     shaper_vec_check(L,Fl,d)
        
    #     y_n = np.ones([L])
    #     y_n[: int(np.sum(d[:1]))] *= z[0]
    #     y_n[int(np.sum(d[:1])) : int(np.sum(d[:2]))] = exponential_space(z[0], z[1], d[1])
    #     y_n[int(np.sum(d[:2])) : int(np.sum(d[:3]))] *= z[1]
    #     y_n[int(np.sum(d[:3])) : int(np.sum(d[:4]))] = exponential_space(z[1], z[0], d[3])
    #     if d[4]> 0:
    #         y_n[int(np.sum(d[:4])): ] = z[0]
            
    #     no_delay_energy = np.sum(y_n**2)
        
    #     # Normalise filter based on weighting with no delay
    #     y *= no_delay_energy/np.sum(y**2)
        
    return y

def shaper_vec_check(L,Fl,d):
    if L < Fl:
            raise ValueError("FIRShaper: Actual filter length is greater than desired filter length")
    if sum(d) != L:
        print("desired length:")
        print(L)
        print("section lengths")
        print(d)
        raise ValueError("FIRShaper: Filter vector sections do not add up")
        
def exponential_space(start,stop,L):
    L = int(L)
    if stop > start:
            x = np.linspace(5,1,L)
            y = np.exp(x)
            # y = x**2
            y *= 1/max(y) 
            y = 1-y
            y = y * (stop - start) + start
            
    elif start > stop:
            x = np.linspace(1,5,L)
            y = np.exp(x)
            # y = x**2
            y *= 1/max(y) 
            y = y * (stop - start) + start

    return y

# === RIR matrix ===
# ------------------------------------------------------------
def rir_matrix(M, L, h, I, J):

    H = np.zeros([M * (J + I - 1), I * L], dtype=np.float32)
    for l in range(L):
        idx_l = slice(l * I, (l + 1) * I)
        for m in range(M):
            idx_m = slice(m * (J + I - 1), (m + 1) * (J + I - 1))

            H[idx_m, idx_l] = Toeplitz(h[:, l, m], I)
    
    return H

# === Toeplitz matrix creation ===
# --------------------------------
def Toeplitz(h, I=[]):
    J = np.size(h)
    if not I:
        I = J
    #  Create zero matrix
    H = np.zeros([J + I - 1, I], dtype=np.float32)
    # H = np.zeros([J + I - 1, I])
    for i in range(I):
        idx = slice(i, np.min([(i + J - 1), J + I - 1]))
        idx2 = slice(i - i, np.min([(i + J - 1), J + I - 1]) - i)
        H[idx, i] = h[idx2]
    return H

# === Time frequency matrix ===
# ------------------------------------------------------------
def time_frequency_matrix(H,M,J,I,NFFT):
    H_reshaped = H.reshape(M, (J + I - 1), -1)  # (M, J+I-1, L*I)
    H_fft = np.fft.fft(H_reshaped, n=NFFT, axis=1).astype(np.complex64)
    H_ = H_fft.reshape(M * NFFT, -1)  # (M*NFFT, L*I)
    
    return H_

# === gamma function crossover ===
# ------------------------------------------------------------
def gamma_crossover_vector(NFFT, Fs, Hz, vals):

    sigma = 0.01
    gamma = np.ones((NFFT))
    Freqs = np.fft.fftfreq(NFFT, d=1/Fs)
    gamma[:NFFT//2] = vals[0] + ((vals[1]-vals[0])/(1+np.exp(-(sigma/2*np.pi)*(Freqs[:NFFT//2]-Hz))))
    gamma[NFFT//2:] = vals[1]

    return gamma

# === Target vector ===
# ------------------------------------------------------------
def target_vector(h_d, NFFT, M, modelling_delay):
    # Creating target vector       
    if M == 1:
        p_target = np.concat([np.zeros([modelling_delay]), h_d])
        P_target = np.fft.fft(p_target, NFFT).astype(np.complex64)
    else:
        P_target = np.zeros([M * NFFT],dtype=np.complex64)
        for m in range(M):
            idx = slice(m * NFFT, (m + 1) * NFFT)
            p_target = np.concat([np.zeros([modelling_delay]), h_d[:, m]])
            P_target[idx] = np.fft.fft(p_target, NFFT)
    return P_target