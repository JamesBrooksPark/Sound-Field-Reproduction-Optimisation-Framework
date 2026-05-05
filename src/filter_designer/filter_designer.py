import numpy as np; import sys
from scipy.sparse import diags, identity as sparse_identity
from scipy.linalg import lu_factor, lu_solve

from src.filter_designer.filter_helper import ConfigLoader, MatrixFormatting, DelayCalculation, FIRShaper, rir_matrix, time_frequency_matrix, gamma_crossover_vector, target_vector

class FilterDesigner():
    def __init__(self, Fs):
        self.Fs = Fs

        # Factors
        self.optimisation_criteria = None # pressure, amplitude, mixed, RoomInRoom, uncomp
        self.delay_supporting = False
        self.ommit_primary = False

        # Initalise global configuration parameters
        # General filter parameters
        self.filter_length = None
        
        # Number of FFT points
        self.NFFT = None
        
        # Delay parameters
        self.modelling_delay = None
        self.precedence_delay = None
        self.calculated_delay_vector = None
        
        # Regularisation parameters
        self.regularisation_weight = None
        self.norm_regularisaion_bool = True
        self.FIR_shaping_weight = None
        
        self.FIR_shaping_norm = True
                    
        # Crossover and gamma values
        self.time_amp_crossover_Hz = None
        self.gamma_vals = None
        
        # Primary source index
        self.primary_source_index = None
        self.supporting_delay = None
        
        # ADMM parameters
        self.rho = None
        self.tolerance_admm = None
        self.maximum_iteration = None
        
        # Global terms to be filled
        self.H_ = None
        self.Hp = None
        self.P_target = None
        self.Ident = None
        self.X = None
        self.X_low = None
        self.X_high = None
        self.main_term = None
        self.M = None
        self.L = None
        self.I = None
        self.J = None
        self.array_type = None
        
         # set base config items from global config file
        config = ConfigLoader()
        for key, value in vars(config).items(): # set self. parameters to the same as the configuration file
            setattr(self, key, value)    
    def filter_designer(self):
        # Check for non filtering methods and return relevant data and 'filter'
        if self.optimisation_criteria == "RoomInRoom":
            print("RoomInRoom")
            sys.stdout.flush() 
            return self.RoomInRoom()
        elif self.optimisation_criteria == "Uncomp":
            print("Uncomp")
            sys.stdout.flush() 
            return self.Uncomp()
            
        
        print("Build core structures")
        sys.stdout.flush() 
        self.build_core_structures()
        
        print("Build matrix")
        sys.stdout.flush() 
        A = self.build_matrix()
            
        print("Solve")
        sys.stdout.flush() 
        w = self.ADMM_solver(A)
        
        
        # Create filter matrix and insert blank filters where required
        if self.L > 1:
            # reshape filter matrix to L x I
            w = w.reshape(self.L, self.I).T
            
        if self.ommit_primary:
            # For supporting filters, insert a modelling delayed delta impulse at the primary source index
            primary_filter = np.zeros((1,self.I))
            if isinstance(self.modelling_delay, int) and 0 <= self.modelling_delay < self.I:
                primary_filter[0, self.modelling_delay] = 1.0
            # Insert the primary filter at the correct index
            w = np.insert(w, self.primary_source_index, primary_filter, axis=1)
        
        meta_data = self.build_meta_data()
        
        return w, meta_data
    
    
    def build_core_structures(self):
        # Format impulse responses and get number of receivers and sources
        h_d,h_p,J = MatrixFormatting(self.target_array_calculation,self.playback_array_calculation)
        
        M = np.shape(h_p)[2]
        L = np.shape(h_p)[1]
        
        I = self.filter_length
        
        self.set_terms(M,L,h_p,h_d, I, J, self.NFFT)
    
    def set_terms(self,M,L,h_p,h_d, I, J, NFFT):
        # Primary source ommission handeling
        if self.ommit_primary:
            L-=1
            # Split primary source from supporting
            h_primary = h_p[:,self.primary_source_index,:]
            h_p = np.delete(h_p, self.primary_source_index, axis=1)
            
        # delay supporting source calculation
        if self.delay_supporting:
            calculated_delay_vector = np.zeros((L))
            if self.ommit_primary:
                for l in range(L):
                    if M > 1:
                        calculated_delay_vector[l] = DelayCalculation(self.Fs,self.precedence_delay,h_primary,h_p[:,l,:])
                    else:
                        calculated_delay_vector[l] = DelayCalculation(self.Fs,self.precedence_delay,h_primary,h_p[:,l])
            else:
                for l in range(L):
                    if l != self.primary_source_index:
                        if M > 1:
                            calculated_delay_vector[l] = DelayCalculation(self.Fs,self.precedence_delay,h_p[:,self.primary_source_index,:],h_p[:,l,:])
                        else:
                            calculated_delay_vector[l] = DelayCalculation(self.Fs,self.precedence_delay,h_p[:,self.primary_source_index],h_p[:,l])
                            
            # Increase filter length by maximum precedence delay
            self.calculated_delay_vector = calculated_delay_vector
            I += int(np.max(self.calculated_delay_vector))
            
            for l in range(L):
                if self.ommit_primary:
                    # if primary is not optimised, all sources are supporting
                    x_l = FIRShaper(I, self.filter_length, self.modelling_delay + calculated_delay_vector[l], 'constant', normalisation= self.FIR_shaping_norm)
                else:
                    # If primary is optimised, primary may have a different filter weight, and no delay
                    if l == self.primary_source_index:
                        x_l = FIRShaper(I, self.filter_length, self.modelling_delay, 'constant', normalisation= self.FIR_shaping_norm)
                    else:
                        x_l = FIRShaper(I, self.filter_length, self.modelling_delay + calculated_delay_vector[l], 'constant', normalisation= self.FIR_shaping_norm) 
                if l == 0:
                    x = x_l
                else:
                    x = np.concatenate((x, x_l))
        else:
            # If delay_supporting is not true, all sources have constant weighting
            x = FIRShaper(I, self.filter_length, self.modelling_delay,'constant', normalisation = self.FIR_shaping_norm)
            x = np.tile(x,L)
                
            self.calculated_delay_vector = None
            
        # create diagonal matrix from filter weighting vector
        X = diags(x, format='csr')
            
        # Create convolutional matrix for playback room rirs
        if self.ommit_primary:
            Hs = rir_matrix(M, L, h_p, I, J)
            Hs_ = time_frequency_matrix(Hs,M,J,I,NFFT)
            
            if M == 1:
                Hp = np.fft.fft(h_primary, NFFT).astype(np.complex64)
            else:
                Hps = np.fft.fft(h_primary, NFFT, axis=0).astype(np.complex64)  
                Hp = Hps.T.reshape(M * NFFT) 
        else:
            H = rir_matrix(M, L, h_p, I, J)
            H_ = time_frequency_matrix(H,M,J,I,NFFT)
            
        # For single method designs, gamma is set to 0 or 1
        if self.optimisation_criteria == 'mixed':
            gamma = gamma_crossover_vector(self.NFFT, self.Fs, self.time_amp_crossover_Hz, self.gamma_vals)
            self.gamma = np.tile(gamma,M)
        if self.optimisation_criteria == 'amplitude':
            self.gamma = 1
        if self.optimisation_criteria == 'pressure':
            self.gamma = 0
        
        # create target vector 
        P_target = target_vector(h_d, NFFT, M, self.modelling_delay)
        
        # Identity matrix for regularisation term
        Ident = sparse_identity(L*I, format='csr')
        
        # Build main term
        if self.ommit_primary:
            main_term =  (Hs_.conj().T @ Hs_).astype(np.complex64)
        else:
            main_term =  (H_.conj().T @ H_).astype(np.complex64)

        if self.ommit_primary:
            self.H_ = Hs_
            self.Hp = Hp
        else:
            self.H_ = H_
        self.P_target = P_target
        self.Ident = Ident
        self.X = X
        self.main_term = main_term
        self.M = M
        self.L = L
        self.I = I
        self.J = J
    
    def build_matrix(self):
        # Number of source dependent regularisation
        if self.norm_regularisaion_bool:
            p = 3
            Lp_norm = np.power(np.sum(np.abs(self.H_)**p), 1/p)
            regularisation_weight = self.regularisation_weight * Lp_norm
        else:
            regularisation_weight = self.regularisation_weight
        
        A = self.main_term.copy()
        if regularisation_weight > 0:
            A += ((2*regularisation_weight)/self.rho) * self.Ident
        if self.FIR_shaping_weight > 0:
                A += ((2*self.FIR_shaping_weight)/self.rho) * self.X

        return A
    
    
    def ADMM_solver(self, A):
        # Extract necessary attributes
        I, L, M, NFFT = self.I, self.L, self.M, self.NFFT
        H_ = self.H_
        P_target = self.P_target
        if self.ommit_primary:
            Hp = self.Hp
        
        # setup solver
        lu, piv = lu_factor(A, overwrite_a=True)
        H_H = H_.conj().T # Hermitian transpose
        rho = self.rho
        
        # Initialising the filter as a delayed delta pulse
        if self.calculated_delay_vector is not None:
            # initial filter containing delayed delta pulses corrosponding with the precedence delay
            w = np.zeros(L*I)
            for l in range(L):
                idx = slice(l * I,(l+1)*I)
                total_delay = int(self.calculated_delay_vector[l]) + self.modelling_delay
                w[idx] = np.concatenate([np.zeros(total_delay),[1],np.zeros(I-total_delay-1)]) # Initialising the filter as a delayed delta pulse
        else:
            w = np.concatenate([np.zeros(self.modelling_delay),[1],np.zeros(I-self.modelling_delay-1)]) 
            w = np.tile(w,L)

        
        # Initialising lagrange multiplier vector 
        lmbda = np.zeros([M * (NFFT)]).astype("complex64")
        
        # Current filter output in frequency domain
        H_w = H_ @ w 
        if self.ommit_primary:
            H_w += Hp

        for ii in range(self.maximum_iteration):
            w_prev = w # Storing the previous filter
            
            h = H_w + lmbda.real / rho
            
            # Calculating desired magnitude + phase
            mag = (np.abs(2 * (1 - self.gamma) * P_target  + self.rho * h) + 2 * self.gamma * np.abs(P_target)) / (self.rho + 2)
            
            phase = (1-self.gamma) * P_target + ((self.rho/2) * h)
            phase = phase / (np.abs(phase))
              
            # Solve for w
            if self.ommit_primary:
                rhs = H_H @ ((mag * phase) - (lmbda / self.rho - Hp))
            else:
                rhs = H_H @ ((mag * phase) - (lmbda / self.rho))
            w = np.real(lu_solve((lu, piv), rhs))

            # Current filter output in frequency domain
            H_w = H_ @ w
            if self.ommit_primary:
                H_w += Hp
            
            # Update lagrange multiplier (error)
            lmbda +=  self.rho * (H_w - mag * phase)
            
            # Display current objective value
            obj = np.linalg.norm(self.gamma*(np.abs(H_w) - np.abs(P_target))) + np.linalg.norm((1-self.gamma)*(H_w - P_target))
            # obj = np.linalg.norm(np.abs(H_w) - np.abs(P_target))
            # print(f"Current amplitude matching value: {obj}")
            
            if ii == 0:
                w_min = w
                obj_min = obj
            elif obj < obj_min:
                w_min = w
                obj_min = obj
            
            # Check convergence
            ddiff = np.linalg.norm(w-w_prev) / np.linalg.norm(w_prev)
            if ddiff <= self.tolerance_admm:
                break
        # print("______________________________________________")
        return w_min
            
    
    def build_meta_data(self):    
        meta_data = {
            'id': 0,
            'N_sources': self.L,
            'N_receivers': self.M,
            'Receiver_Geom': self.array_type,
            'method': self.optimisation_criteria,
            'delay_supporting': self.delay_supporting,
            'ommit_primary': self.ommit_primary,
            'filter_length': self.I,
            'NFFT': self.NFFT,
            'modelling_delay': self.modelling_delay,
            'regularisation_weight': self.regularisation_weight,
            'FIR_shaping_weight': self.FIR_shaping_weight,
            'gamma_value_lower': self.gamma_vals[0],
            'gamma_value_higher': self.gamma_vals[1],
            'gamma_crossover': self.time_amp_crossover_Hz
        }
        return meta_data


    def RoomInRoom(self):
        # w = np.mean(self.target_array_calculation, axis=1)
        w = self.target_array_calculation[:,0]
        # M = np.shape(self.d_arr)[1]
        M = 1
        meta_data = {
            'id': 0,
            'N_sources': 1,
            'N_receivers': M,
            'Receiver_Geom': self.array_type,
            'method': "RoomInRoom",
            'delay_supporting': self.delay_supporting,
            'ommit_primary': self.ommit_primary,
            'filter_length': len(w),
            'NFFT': 0,
            'modelling_delay':0,
            'regularisation_weight': 0,
            'FIR_shaping_weight': 0,
            'gamma_value_lower': 0,
            'gamma_value_higher': 0,
            'gamma_crossover': 0
        }
        return w, meta_data
    
    def Uncomp(self):
        I = self.filter_length
        w = np.zeros(I)
        w[0] = 1
        M = 1
        meta_data = {
            'id': 0,
            'N_sources': 1,
            'N_receivers': M,
            'Receiver_Geom': self.array_type,
            'method': "Uncomp",
            'delay_supporting': self.delay_supporting,
            'ommit_primary': self.ommit_primary,
            'filter_length': I,
            'NFFT': 0,
            'modelling_delay':0,
            'regularisation_weight': 0,
            'FIR_shaping_weight': 0,
            'gamma_value_lower': 0,
            'gamma_value_higher': 0,
            'gamma_crossover': 0
        }
        return w, meta_data
        