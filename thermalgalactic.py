#!/usr/bin/env python

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import random
import sys 
import scipy.signal.signaltools as sigtool
import matplotlib.pylab as pylab
from matplotlib.ticker import FuncFormatter


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

def rfftfreq(n, d=1.0, nyquist_domain=1):
    '''calcs frequencies for rfft, exactly as numpy.fft.rfftfreq, lacking that function in my old numpy version.
    Arguments:
    ---------
        n: int
            Number of points.
        d: float
            Sample spacing, default is set to 1.0 to return in units of sampling freq.
        
    Returns:
    -------
        f: array of floats
            frequencies of rfft, length is n/2 + 1
    '''
    if n % 2 == 0:
        f = array([n/2 - i for i in range(n/2,-1,-1)]) / (d*n)
    else:
        f = array([(n-1)/2 + 1 - i for i in range(n/2,-1,-1)]) / (d*n)
    # if nyquist_domain is 1 you're done and return directly
    if nyquist_domain != 1:
        # if nyquist_domain even, mirror frequencies
        if (nyquist_domain % 2) == 0: f = f[::-1]
        sampling_freq = 1./d
        fmax = 0.5*sampling_freq
        f += (nyquist_domain-1)*fmax
    return f

AERA=True
print "AERA",AERA

imp = 376.7303 ##vacuum impedance
kb = 1.38064852e-23##boltzmann constant
n = 2048# ####put the size of your time series here

dt = 4.162e-07/float(n)###put required dt here. i.e., same as your signal simulations. If you put incorrect dt, you will get skewed results
fsample = 1./dt###sampling rate

#stnnum=str(45)
stnnum=str(54)
stntype='EW'
#~ stntype='NS'	


##Defining the noise model
def noise_func(f):		##freq in MHz
	Ig = 2.48*(10**-20)
	Ieg = 1.06 * (10**-20)
	tau = 5.0 * (f**-2.1) #checked
	term1 = Ig *(f**-0.52) *(1. - np.exp(-tau))/tau
	term2 =  Ieg *(f**-0.80) * np.exp(-tau)
	I = (term1 + term2)#*(10**3)#makes it mW.. ##in W/Hz/sr/m^2
	return I		##Noise in brightness


def calculation(filename):	###SNR calculation, hand in filename of file containing filtered Ex,Ey,Ez after response (coreas style)
	f1 = open(filename,'r')
	freq=[]
	
	Power=[]

	c = (3.e8) ##speed of light in m/s
	
	thermal = 300.		##Thermal component of noise from electronics in Kelvin
	#~ thermal = 40.		##Thermal component of noise from electronics in Kelvin
	#print "THIS IS SIMULATED WITH 300 K THERMAL NOISE"
	
	
	nf = n+1###giving in odd numbered time series size will avoid weird kinks in the spectrum
	freq1 = (rfftfreq(nf, dt))##in Hz
	
	df = abs(freq1[0] - freq1[1])
	Temp = []
	B = []
	Therm = []

	farea = open('areavsfreq150MHz.dat','r')		###antenna file with effective area, needed for noise, this is integrated response over theta and phi
	freq =[]
	eff_area=[]
	# ~ eff_height=[]
	# ~ impedance=[]
	for line in farea:
		if not line.strip().startswith("#"):
			columns = line.split()		
			#print columns
			
			eff_area.append(float(columns[1]))
			# ~ eff_height.append(float(columns[2]))###not needed
			# ~ impedance.append(float(columns[3]))

				
	eff_area = np.array(eff_area)

	#print "area",eff_area.size
	neff=0
	
	for i in range (0,len(freq1)):
		y = freq1[i]
		
		freq.append(y)
		if y<=fmax and y>=fmin:###defines freq band we want in Hz
			y = y/(10**6)##convert to MHz since noise func needs freq in MHz
			x = noise_func(y)
			
			lamb = c/freq1[i]
			
			Brightnesstemp=(x*lamb*lamb/(2.*kb))				###calculates temperature of galactic noise
			
			B.append(Brightnesstemp)
			Therm.append(thermal)
			Temptot = (Brightnesstemp+thermal)
			Temp.append(Temptot)
			
			Power.append(kb*Temptot*df*2*eff_area[neff])				###Power in each freq bin
					
		
		else:
			y = y/(10**6)
			x = noise_func(y)
						
			lamb = c/freq1[i]
			
			Brightnesstemp=(x*lamb*lamb/(2*kb))
			
			B.append(Brightnesstemp)
			Therm.append(thermal)
			Temptot = (Brightnesstemp+thermal)
			Temp.append(Temptot)
			
			Power.append(0)			### Noise outside band set to zero
			
		neff = neff+1
	del freq1
	freq = np.array(freq)
	##Integrating out the solid angle
	
	Power = (np.array(Power))*0.5###in W		######Because of half polarization

	df = abs(freq[0] - freq[1])
	
	Amplitude = (np.sqrt(Power*2*imp))##in V/m						####Power to amplitude in freq space
	
	Amplitude=Amplitude.astype(complex64)##Converting to complex type
	
	norm = Amplitude.size		####Required normalization according to numpy
	
	##Adding phase infor to noise amplitude 
	
	for i in range (0,len(Amplitude)):
		phi = (2*np.pi*random.random())#*i*freq[i]/norm)
		
		#Amplitude[i] = Amplitude[i]*exp(-1j*phi)*np.sqrt(2*norm)#####Normalization originally used for paper
		Amplitude[i] = Amplitude[i]*exp(-1j*phi)*(2*norm)#####Normalization for erratum
		
		if Amplitude[i]!=0:
			Amplitude[i] = Amplitude[i] *(2*(73.2/imp)**0.5)####Corrected (4*(73.2/imp)**0.5) to (2*(73.2/imp)**0.5)###73.2 is avg impedence for dipole antenna


	totnoisepow = ((Power[0]) +(2 * np.sum(Power[1:])))#
	
	print "total in freq domain", totnoisepow
####################Reading in signal###############
	time = []
	sigtime=[]
	
	eX=[]
	eY=[]
	eZ=[]
	for line in f1:
		if not line.strip().startswith("#"):
			columns = line.split()
			t = float(columns[0]) ##time information 
			E1 = float(columns[1]) ####This is how coreas writes out file. ZHAires uses the opposite order
			E2 = float(columns[2])
			E3 =  float(columns[3])
						
			eX.append(E1)
			eY.append(E2)
			eZ.append(E3)
				
			sigtime.append(t)
	
	
	Xenv = np.abs(sigtool.hilbert(eX)) # hilbert(s) actually returns the analytical signal
	Yenv = np.abs(sigtool.hilbert(eY)) # hilbert(s) actually returns the analytical signal
	eX = np.array(eX)
	eY = np.array(eY)
	
	Xmaxenv = np.amax(Xenv)
	Ymaxenv = np.amax(Yenv)
	
	t=sigtime[0]
	for k in range(0,n):
		time.append(t)
		t=t+dt
	nt = len(time)
	
	##Amplitude of noise in time domain
	tAmp = (np.fft.irfft(Amplitude))
	
	#print "lengths",len(tAmp), len(Amplitude)
	tPowN = tAmp*tAmp#/(73.2)
	totnoisepowtime = np.sum(tPowN)
	
	print "total in time dom", totnoisepowtime
	

	Noiseenv = np.abs(sigtool.hilbert(tAmp))
	Noisemaxenv = (10**6)*np.amax(Noiseenv)
	maxpownoise = np.amax(tPowN)
	
	###Uncomment to plot brightness##
	# ~ Xplt = np.linspace(1,500,5000)
	# ~ Yplt = noise_func(Xplt)
	# ~ fig3=plt.figure(figsize=(7,4.85),facecolor="white")
	# ~ ax3 = fig3.add_subplot(111)
	# ~ ax3.plot(Xplt,Yplt,label = 'Brightness')
	# ~ ax3.set_xlim([1,400])
	# ~ ax3.set_ylim([10**(-21),10**(-19)])
	# ~ xlabels=[0.1,1,10,100,400,500]
	
	
	# ~ ax3.loglog()
	# ~ ax3.set_xticklabels(xlabels, ha='center')
	# ~ ax3.yaxis.labelpad = 0.0001
	# ~ ax3.set_ylabel('Brightness(Wm$^{-2}$Hz$^{-1}$sr$^{-1}$)',va='center',fontsize=15,labelpad = 10)
	# ~ ax3.set_xlabel('Frequency (MHz)',va='center',fontsize=15,labelpad = 10)
	
	# ~ fig3.tight_layout(w_pad=0.5,h_pad = 1.2)
	
	# ~ del Xplt,Yplt
	########################################Calculating RMS noise######################################
	
	rmsnoise = 0.
	for kj in range(0,tAmp.size):
		rmsnoise = rmsnoise + (tAmp[kj]*tAmp[kj])
	rmsnoise = (10**6)*((rmsnoise/tAmp.size)**0.5) ##########microvolt

	SNRx = (Xmaxenv*Xmaxenv)/(rmsnoise*rmsnoise)
	SNRy = (Ymaxenv*Ymaxenv)/(rmsnoise*rmsnoise)
	
	return B,Temp,Power,sigtime,tAmp,eX,eY,eZ,Xenv,freq,Therm,Amplitude,time,SNRx,SNRy
	
if (__name__ == '__main__'):
	fmin = int(30.e6)
	fmax = int(80.e6)
	####Hand over a test signal file which already has antenna response###
	filenamelist=["/cr/data01/balagopal/sim_data/reas_gnp/trial/210106/SIM210106_stn45_difffilter/150MHzresponse_timeseries_filtered_stn45_50-350.dat"]
	filename =filenamelist[0] 
	#f1 = open(filename,'r')
	#~ fig=plt.figure(facecolor="white",figsize=(13,4))####to plot noise traces
	fig=plt.figure(facecolor="white",figsize=(6,5))
	fig2=plt.figure(facecolor="white")
	
	fig.text(0.49, 0.01, 'Time (ns)', ha='center',fontsize=17)
	#fig.text(0.04, 0.5, 'Power (mW m$^{-2}$ MHz$^{-1}$)', va='center', rotation='vertical',fontsize=15)
	#~ fig.text(0.01, 0.5, 'Field strength($\mu$V)', va='center', rotation='vertical',fontsize=15)
	fig.text(0.001, 0.51, 'Amplitude ($\mu$V)', va='center', rotation='vertical',fontsize=17)

	#~ fig2.text(0.5, 0.01, 'Frequency (MHz)', ha='center',fontsize=15)
	#~ #fig.text(0.04, 0.5, 'Power (mW m$^{-2}$ MHz$^{-1}$)', va='center', rotation='vertical',fontsize=15)
	#~ fig2.text(0.01, 0.5, 'Noise temperature (K)', va='center', rotation='vertical',fontsize=15)
	
	B,Temp,Power,sigtime,tAmp,eX,eY,eZ,Xenv,freq,Therm,Amplitude,time,SNRx,SNRy=calculation(filename)
	
	fig4=plt.figure(figsize=(7,4.85),facecolor="white")
	ax4 = fig4.add_subplot(111)
	df = freq[1]-freq[0]
	
	print(SNRx,SNRy)
	
	
	time = np.array(time)
        print(tAmp.max(),eX.max(),eY.max())
	sigtime = np.array(sigtime)
	ax = fig.add_subplot(111)
	ax.tick_params(labelsize=15)

	l2 = ax.plot(time[:]/(10**-9),tAmp*(10**6),'--b',label='Noise')

	# ~ l3 = ax.plot(time[:]/(10**-9),eX,'-',color='r',label='Signal')

	plt.setp(l2, linewidth=1.4)
	ax.set_ylim(-82,56)

	ax.axhline(0, color='black')


	ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

	ax2 = fig2.add_subplot(111)
	ax2.plot(freq[:]/(10**6),Temp,label = 'Total')
	ax2.plot(freq[:]/(10**6),B,'--g', label = 'Galactic')
	ax2.plot(freq[:]/(10**6),Therm,'--r' ,label = 'Thermal')
	ax2.set_xlim([0,400])
	ax2.semilogy()
	ax2.set_ylim(bottom =10)
	ax2.set_ylabel('Noise temperature (K)',va='center',fontsize=15,labelpad = 10)
	ax2.set_xlabel('Frequency (MHz)',va='center',fontsize=15,labelpad = 10)
	fig2.tight_layout(w_pad=0.5,h_pad = 1.2)
	
	
	
	ax2.legend(loc='best', fancybox=True)
	ax.legend(loc='best', fancybox=True,fontsize=16)
	print SNRx,SNRy
	#~ plt.show()
	#fig.savefig('teststuff2.png')
	del B,Temp,Power,sigtime,tAmp,eX,eY,eZ,Xenv,freq,Therm,Amplitude,time,SNRx,SNRy
	#~ del B,Temp,PowerperHz,sigtime,tAmp,eX,eY,SigEW,Xenv,EWenv,freq,Therm,Amplitude,eff_area,eff_height,time,tPowN
	del SNRx,SNRy
else:
		
	from SNR import files
	filenamelist,num= files()
	#~ fmin = int(10**6*fmin)
	#~ fmax = int(10**6*fmax)
	##print filenamelist
	for index, item in enumerate(filenamelist):
		filename =item 
		print "WARNING!!! frequencies hardcoded"
		#~ fmin = int(50.e6)
		#~ fmax = int(350.e6)
		fmin = int(100.e6)#####Give freq band here
		fmax = int(190.e6)
		#~ fmin = int(30.e6)
		#~ fmax = int(80.e6)
                ####Uncomment to scan over freq ranges###
		#for i,c in enumerate(filename):
			#if c=="-"and filename[i-3]=="_":
			#	fmin = int(filename[i-2:i])
			#	fmax = int(filename[i+1:-4])
				#~ #print "from file", fmin,fmax
			#	fmin = int(10**6*fmin)
			#	fmax = int(10**6*fmax)
			#elif c=="-"and filename[i-4]=="_":
			#	fmin = int(filename[i-3:i])
			#	fmax = int(filename[i+1:-4])
				#~ #print "from file", fmin,fmax
			#	fmin = int(10**6*fmin)
			#	fmax = int(10**6*fmax)
				
		#~ num = str(115)
		#~ stnnum= str(index+1+2364)
		stnnum= str(index+1)
		#stnnum=str(54)
		#~ stnnum=str(49)
		#stnnum=str(45)
		stntype='EW'
		
		B,Temp,Power,sigtime,tAmp,eX,eY,eZ,Xenv,freq,Therm,Amplitude,time,SNRx,SNRy=calculation(filename)
		
		Xmaxenv = np.amax(Xenv)
		#print Xmaxenv
		del B,Temp,Power,sigtime,tAmp,eX,eY,eZ,Xenv,freq,Therm,Amplitude,time
		
		print "WARNING!! files open in append mode"
		#path='/cr/data01/balagopal/FromHome//SNRvalues/Erratum/'
		#with open(path+"SNR_stn"+stnnum+"_SIM"+num+"_150MHzEWresonance_response.dat", "a") as myfile:
		path=''##If you want a path for your SNR files, put it here
		with open(path+"SNR_SIM"+num+"_"+str(fmin/10**6)+"-"+str(fmax/10**6)+".dat", "a") as myfile:

			myfile.write(stnnum+"\t"+str(SNRx)+"\t"+str(SNRy)+"\n")	
		del SNRx,SNRy
