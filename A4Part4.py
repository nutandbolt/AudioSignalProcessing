import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

"""
A4-Part-4: Computing onset detection function (Optional)

Write a function to compute a simple onset detection function (ODF) using the STFT. Compute two ODFs 
one for each of the frequency bands, low and high. The low frequency band is the set of all the 
frequencies between 0 and 3000 Hz and the high frequency band is the set of all the frequencies 
between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 

A brief description of the onset detection function can be found in the pdf document (A4-STFT.pdf, 
in Relevant Concepts section) in the assignment directory (A4). Start with an initial condition of 
ODF(0) = 0 in order to make the length of the ODF same as that of the energy envelope. Remember to 
apply a half wave rectification on the ODF. 

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a numpy 
array with two columns, where the first column is the ODF computed on the low frequency band and the 
second column is the ODF computed on the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases. For test case 1, you can clearly see that the ODFs have sharp peaks at the onset of the 
piano notes (See figure in the accompanying pdf). You will notice exactly 6 peaks that are above 
10 dB value in the ODF computed on the high frequency band. 
"""

def computeODF(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF 
            computed on the low frequency band and the second column is the ODF computed on the high 
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz 
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """
    
    ### your code here
     
    ### your code here
    if M%2  :       #Ensuring M IS ODD
        M=M-1   
        
    fs, x = UF.wavread(inputFile)
    
    w = get_window(window, M)
    
    mX, pX = STFT.stftAnal(x, w, N, H)
    
    print("mX Shape",mX.shape)
    
    #convert to linear scale 
    mX_linear=np.power(10,(mX/20.0))
    
    frames=mX.shape[0] # mX is number of frames X 1/2 fft size
    #print("frames",frames)
    
    freq=fs*np.arange(0,N/2+1)/N   # 1 added since number of points includes 0
    time=np.arange(0,frames*H,H)/fs 
    low_band,=np.where((freq>0) & (freq<3000)) # returns a tuple hence the , on LHS
    high_band,=np.where((freq>3000)&(freq<10000)) 
    high_energy=[]
    low_energy=[]
    odf_low_energy=[]
    odf_high_energy=[]
    odf_low_energy.append(0.0)
    odf_high_energy.append(0.0)
    
    #print(low_band.size,high_band.size)
    
    
    for frame in range(0,frames): 
        
        frame_spectra=mX_linear[frame,:]
        #print(frame_spectra.shape) 
        low_energy.append(sum(frame_spectra[low_band]**2))
        high_energy.append(sum(frame_spectra[high_band]**2))
        
    low_energy_db=10*np.log10(low_energy) 
    high_energy_db=10*np.log10(high_energy)  
    #engEnv=np.array(low_energy_db,high_energy_db)
    
    for frame in range(1,frames):
    
        odf_low_energy.append(low_energy_db[frame]-low_energy_db[frame-1])
        odf_high_energy.append(high_energy_db[frame]-high_energy_db[frame-1])
        
    odf_low_energy=np.array(odf_low_energy)
    odf_high_energy=np.array(odf_high_energy)
    odf_low_energy[odf_low_energy < 0.0]=0.0
    odf_high_energy[odf_high_energy < 0.0]=0.0  
      
    
    plt.subplot(2,1,1)
    plt.pcolormesh(time, freq, np.transpose(mX))
    plt.axis([0,4.0,0,20000])
    plt.subplot(2,1,2)
    plt.plot(time,odf_low_energy)
    plt.plot(time,odf_high_energy)
    #plt.plot(time,low_energy_db)
    #plt.plot(time,high_energy_db)
    #plt.axis([0,4.0,-80,-10])
    plt.show()
    return np.column_stack((odf_low_energy,odf_high_energy))
      