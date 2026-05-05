import numpy as np

class Smoothing:
# Smoothing code taken from
#  https://github.com/Panagiotis-Zachos/complex_smoothing/tree/main
# Which is the official implementation from the paper:
#  Hatziantoniou, P. D., & Mourjopoulos, J. N.: Generalized fractional-octave smoothing of audio and acoustic responses.
# Python Implementation:
# James Brooks-Park
# James.Brooks-Park@Uni-Oldenburg.de

    def __init__(self, NFFT=0):
        self.NFFT = NFFT

    def complexSmoothing(self, irIn, smoothingMethod, smoothFactor, oct):
        #    Function complexSmoothnig
        #    created by Panagiotis (Panos) Hatziantoniou
        #    updated by Panagiotis Zachos
        #    Python implementation by James Brooks-Park
        #    Outputs:  - irSmoothed: Time domain impulse respone after spectral smoothing (samples)
        #              - specOut: Smoothed Spectrum (complex samples)
        #    Inputs:   - irIn: Time domain original impulse response (samples) (Numpy array)
        #              - smoothMethod: Parameter for smoothing method selection.
        #                             'spectrum' : smoothing on spectral magnitude (meter)
        #                             'power'       : smoothing on Power spectrum
        #                             'db'    : smoothing on spectral dB scaled magnitude
        #                             'phase'    : smoorthing only on phase
        #                             'complex'  : Complex Smoothing i.e. smoothng on spectral real part
        #                                          and imaginary part
        #                             'mixed'    : Mixed complex smoothing, i.e. building the
        #                                          spectrum from smoothing, seperately, the
        #                                          magnitude and the phase
        #            - smoothFactor: Defines the shape of the smoothing (averaging)
        #                        moving window along the frequency bins. 1 corresponds to a
        #                        rectagular window, values < 1 correspond to weighted averaging
        #            - oct: The octave fraction (examples: 1 for 1-octave smothing, 0.33 for 1/3-octave)
        #

        """Error Checking"""
        if type(irIn) is not np.ndarray:
            raise ValueError("Input impulse response is not of type Numpy array")

        if smoothingMethod not in [
            "spectrum",
            "power",
            "db",
            "phase",
            "complex",
            "mixed",
        ]:
            raise ValueError(
                "Incorrect smoothing method, correct methods are as follows:\n spectrum \n power\n db\n phase\n complex\n mixed\n"
            )

        if not isinstance(smoothFactor, (float, int)):
            raise ValueError("Soothing factor is not numerical")

        if not isinstance(oct, (float, int)):
            raise ValueError("Octave factor is not numerical")

        """Function"""
        sh = np.size(irIn)
        if self.NFFT == 0:
            NFFT = self.nextpow2(sh)
        else:
            NFFT = self.NFFT

        Fn = int(NFFT / 2 + 1)

        IRIn = np.fft.fft(irIn, NFFT)
        specIn = abs(IRIn)
        phaseIn = np.angle(IRIn)

        specOut = np.zeros(NFFT, dtype="cdouble")

        if smoothingMethod == "spectrum":
            temp1 = abs(specIn)
            winSizes = self.PsychoModel(Fn, oct)
            out = self.smoothData(temp1, Fn, winSizes, smoothFactor)
            specOut[0:Fn] = (out * np.cos(phaseIn[0:Fn])) + 1j * (
                out * np.sin(phaseIn[0:Fn])
            )

        if smoothingMethod == "power":
            temp1 = (abs(specIn)) ** 2
            winSizes = self.PsychoModel(Fn, oct)
            out = self.smoothData(temp1, Fn, winSizes, smoothFactor)
            specOut[0:Fn] = (
                np.sqrt(np.transpose(out)) * np.cos(phaseIn[0:Fn])
            ) + 1j * (np.sqrt(np.transpose(out)) * np.sin(phaseIn[0:Fn]))

        if smoothingMethod == "db":
            temp1 = 20.0 * np.log10(abs(specIn))
            winSizes = self.PsychoModel(Fn, oct)
            out = self.smoothData(temp1, Fn, winSizes, smoothFactor)
            specOut[0:Fn] = 10.0 ** (np.transpose(out) / 20.0) * np.cos(
                phaseIn[0:Fn]
            ) + 1j * (10.0 ** (np.transpose(out) / 20.0) * np.sin(phaseIn[0:Fn]))

        if smoothingMethod == "phase":
            temp1 = abs(specIn)
            winSizes = self.PsychoModel(Fn, oct)
            out = self.smoothData(phaseIn, Fn, winSizes, smoothFactor)
            out = np.transpose(out)
            specOut[0:Fn] = (temp1[0:Fn] * np.cos(out[0:Fn])) + 1j * (
                temp1[0:Fn] * np.sin(out[0:Fn])
            )

        if smoothingMethod == "complex":
            winSizes = self.PsychoModel(Fn, oct)
            Rsm = self.smoothData(np.real(specIn), Fn, winSizes, smoothFactor)
            Ism = self.smoothData(np.imag(specIn), Fn, winSizes, smoothFactor)
            specOut[0:Fn] = Rsm[0:Fn] + 1j * Ism[0:Fn]

        if smoothingMethod == "mixed":
            temp1 = abs(specIn)
            winSizes = self.PsychoModel(Fn, oct)
            out = self.smoothData(temp1, Fn, winSizes, smoothFactor)
            Rsm = self.smoothData(np.real(specIn), Fn, winSizes, smoothFactor)
            Ism = self.smoothData(np.imag(specIn), Fn, winSizes, smoothFactor)
            specOut[0:Fn] = out[0:Fn] * (
                np.cos(np.angle(Rsm[0:Fn] + 1j * Ism[0:Fn]))
                + 1j * (np.sin(np.angle(Rsm[0:Fn] + 1j * Ism[0:Fn])))
            )

            specOut[Fn:NFFT] = np.flip(np.real(specOut[2:Fn])) + 1j * (-1) * np.flip(
                np.imag(specOut[2:Fn])
            )

        irSmoothed = np.real(np.fft.ifft(specOut))
        return irSmoothed

    def smoothData(self, datIn, datLen, win_sizes, smoothFactor):
        # arguments
        #     datIn (1,:) {mustBeNumeric}
        #     datLen (1,1) {mustBePositive}
        #     winSizes (1,:) {mustBeNumeric}
        #     smoothFactor double {mustBeInRange(smoothFactor, 0, 1)} = 0.99
        # end

        out = np.zeros(int(datLen))

        for ii in range(datLen):
            if win_sizes[ii] > 1:
                smoothWin = self.smoothingWindow(win_sizes[ii] * 2, smoothFactor)

                StartId = int(ii - win_sizes[ii])
                EndId = int(ii + win_sizes[ii])

                if StartId < 1:
                    smoothWin = smoothWin[-StartId:-1]
                    StartId = 1
                if EndId > datLen:
                    smoothWin = smoothWin[0 : -1 - (EndId - datLen - 1)]
                    EndId = datLen

                out[ii] = sum(datIn[StartId:EndId] * smoothWin) / (
                    2 * win_sizes[ii] + 1
                )
            else:
                out[ii] = datIn[ii]

        return out

    def PsychoModel(self, Fn, oct):
        # arguments
        # len {mustBePositive}
        # oct double {mustBePositive, mustBeLessThan(oct, 1)}
        # end

        larg = (np.sqrt(2.0) ** oct) - (np.sqrt(1 / 2) ** oct)
        win_sizes = np.round(0.5 * larg * np.linspace(1, Fn, Fn))

        return win_sizes

    def smoothingWindow(self, win_size, smoothFactor):
        # arguments
        #     smoothFactor double {mustBeInRange(smoothFactor, 0, 1)}
        # end
        if win_size <= 0:
            raise ValueError("Window size must be positive")

        factor = 8.0 * np.arctan(1) / (win_size - 1)
        win = smoothFactor - (1.0 - smoothFactor) * np.cos(
            factor * np.linspace(1, win_size - 1, int(win_size))
        )
        return win

    def Phase(self, In):
        return np.arctan2(np.imag(In), np.real(In))

    def nextpow2(self, N):
        """Function for finding the next power of 2"""
        n = 1
        while n < N:
            n *= 2
        return n