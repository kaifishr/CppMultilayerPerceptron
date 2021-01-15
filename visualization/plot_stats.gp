#!/usr/bin/gnuplot

#############
# Plot loss #
#############

reset

x_size = 384
y_size = 512
set terminal pngcairo size y_size,x_size enhanced font "Verdana,9"
datafile = "../results/stats.dat"

set output 'loss.png'

set grid

set xlabel 'Epochs'

#plot datafile every ::0::100 u (10 * $0):1 w l ls 2 lc "black" title "loss"
plot datafile u (10 * $0):1 w l ls 2 lc "black" title "loss"

#################
# Plot accuracy #
#################

#reset

set output 'accuracy.png'

set grid

set xlabel 'Epochs'

#plot datafile every ::0::100 u (10 * $0):2 w l ls 2 lc "black" title "accuracy"
plot datafile u (10 * $0):2 w l ls 2 lc "black" title "accuracy"
