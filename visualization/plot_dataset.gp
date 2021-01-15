#!/usr/bin/gnuplot

#################################
# Visualize predictionlandscape #
#################################

reset

set terminal pngcairo size 512,512 enhanced font 'Verdana,10'
set output 'prediction_landscape.png'

unset key
unset colorbox
load "plasma.pal"

#set xlabel 'x'
#set ylabel 'y'

set size ratio -1
unset key
unset xtics
unset ytics

datafile_0 = "../results/prediction_landscape.dat"
datafile_1 = "../results/dataset.dat"

stats datafile_0

point_size = 0.5

plot \
datafile_0 u (2*(($1)/STATS_records)-1):(2*(($2)/STATS_records)-1):($3) matrix with image, \
#datafile_1 u ($1):2:3 pt 7 ps point_size palette, \
#datafile_1 u ($1):2:3 pt 6 ps point_size lc "black" lw 0.2
#datafile_1 u (-$1):2:3 pt 7 ps point_size palette, \
#datafile_1 u (-$1):2:3 pt 6 ps point_size lc "black" lw 0.2

#####################
# Visualize dataset #
#####################

reset

set terminal pngcairo size 512,512 enhanced font 'Verdana,10'
set output 'dataset.png'

unset key
unset colorbox

set xlabel 'x'
set ylabel 'y'

set size ratio -1
unset key
set grid
#unset xtics
#unset ytics

plot datafile_1 u ($1):2:3 pt 7 ps point_size palette, \
     datafile_1 u ($1):2:3 pt 6 ps point_size lc "black" lw 0.1
#plot datafile_1 u (-$1):2:3 pt 7 ps point_size palette, \
#     datafile_1 u (-$1):2:3 pt 6 ps point_size lc "black" lw 0.1

###########################################
# Visualize discrete prediction landscape #
###########################################

reset

set terminal pngcairo size 512,512 enhanced font 'Verdana,10'
set output 'prediction_landscape_discrete.png'

unset key
unset colorbox

#set xlabel 'x'
#set ylabel 'y'

set size ratio -1
unset key
unset xtics
unset ytics

set palette maxcolors 2
set palette defined (0 '#0c0887', 1 '#f0f921')
#set multiplot
plot datafile_0 u (2*(($1)/STATS_records)-1):(2*(($2)/STATS_records)-1):3 matrix with image, \
#datafile_1 u ($1):2:3 pt 7 ps point_size palette, \
#datafile_1 u ($1):2:3 pt 6 ps point_size lc "black" lw 0.2
#datafile_1 u (-$1):2:3 pt 7 ps point_size palette, \
#datafile_1 u (-$1):2:3 pt 6 ps point_size lc "black" lw 0.2
