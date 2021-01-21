import numpy as np
import thermalgalactic
import sys
from sys import argv


filenamelist=[]


script = sys.argv[0]
num = sys.argv[1]###The number of your simulation. Here is is 613429

stnnum=str(1)

print num


for i in range(int(stnnum),int(stnnum)+2):###Put in number of stations you have## 
	
	toutfilename = "SIM"+num+"_Sample_station_number_"
	
	####Use this to scan over all files with same freq range
	filename = toutfilename + str(i)+".dat"
	filenamelist.append(filename)
	def files():
		#print filenamelist
		return filenamelist,num
