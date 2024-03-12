#!/usr/bin/env bash

set -xeu

if ! command -v "gnuplot" > /dev/null ; then
    echo "ERROR: gnuplot is not installed!"
    exit 1
fi

# convert file
if ! test -e 'data.txt'; then
    echo "data.txt not found!"
    exit 1
fi
awk -F'[]]|[[]' '/------/{printf "\n"$2" "; next} {printf $2" "}' data.txt | tail -n +2 > data.dat
if ! test -e 'data.dat'; then
    echo "data.dat not found!"
    exit 1
fi

# plot
gnuplot -e '
set datafile separator whitespace;
set output "output.png";
set terminal png size 2000,600;
set multiplot layout 3,2;
plot "data.dat" using 1:2 with linespoints title "TA\\_BUSY\\_avr";
plot "data.dat" using 1:3 with linespoints title "CU\\_OCCUPANCY";
plot "data.dat" using 1:4 with linespoints title "CU\\_UTILIZATION";
plot "data.dat" using 1:5 with linespoints title "TOTAL\\_16\\_OPS";
plot "data.dat" using 1:6 with linespoints title "TOTAL\\_32\\_OPS";
'

# show image in terminal if supported
if command -v "fish" > /dev/null ; then
    fish -c 'type -q icat; and icat output.png'
fi

echo "Output file: output.png"
