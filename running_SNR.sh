#!/bin/bash
#$ -S /bin/bash

##cd diffenergyrange/
##cd diffenergyrange/
for inpnum in `seq 613429 613430`;do
#~ for inpnum in `seq 280699 280703`;do
#~ for inpnum in `seq 610268 610268`;do
#~ for inpnum in `seq 6101 6109`;do
#for inpnum in `seq 210108 210115`;do
#for inpnum in `seq 210106 210106`;do
#~ for inpnum in `seq 471 479`;do
#~ for inpnum in `seq 611 619`;do
#~ for inpnum in `seq 321 329`;do
#~ for inpnum in `seq 311 319`;do
    fnum=${inpnum}
    python SNR.py $fnum
done
##for inpnum in `seq 469 469`;do
##      fnum=${inpnum}420
##      echo $fnum
##      mkdir ${fnum}
##      mv DAT${fnum} DAT${fnum}.long ${fnum}/
##      mv SIM${fnum}* ${fnum}/
##done
