#!/bin/bash

# Convert SVG output from matplotlib to PDF using Inkscape.
#
# Why not just save directly to PDF from matplotlib in the first place? Because
# there are bugs or limitations with matplotlib saving to PDF, specifically
# with regards to fonts (some special characters are converted to paths rather
# than embedded as text*) and the figure layout (it does not seem to precisely
# match the interactively-displayed figure which is used to test the layout).
# Exporting initially to SVG, and then using Inkscape to save PDF copies,
# avoids these issues.
#
# *known bug: github.com/matplotlib/matplotlib/issues/21797
#
# First argument passed to script is the directory of the output figures
#
# ----------------------------------------------------------- # 

figDirIn=${1}
figDirOut=$figDirIn

if [[ -d $figDirIn ]]
then
    for figIn in $(ls $figDirIn/*.svg)
    do
        figOut="${figIn%.*}".pdf
        inkscape $figIn --export-type=pdf --export-filename=$figOut
        echo "Saved ${figOut}"
    done
else
    echo "Error: directory ${figDirIn} does not exist."
    exit 2
fi

exit 0

