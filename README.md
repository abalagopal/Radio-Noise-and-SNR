# Radio-Noise-and-SNR

The main noise code "thermalgalactic.py". You can call signal files (with antenna response) through SNR.py. I have provided 2 sample signal files.
These are "SIM613429_Sample_station_number_1.dat" and "SIM613429_Sample_station_number_2.dat"

You can run the code as "python SNR.py 613429"
where 613429 is the name of the simulation. Running this will produce the output file SNR_SIM613429_100-190.dat

I have also provided a file named "areavsfreq150MHz.dat". This is the file that is passed on to the script "thermalgalactic.py" so that the effective area of the antenna can be folded into the noise script (to get noise in terms of V instead of V/m).
