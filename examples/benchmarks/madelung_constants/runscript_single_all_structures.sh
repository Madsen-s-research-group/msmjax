#!/bin/bash

python calculate_madelung_single.py -p 6 --structure NaCl-conventional --outdir out/single/NaCl-conventional/
python calculate_madelung_single.py -p 6 --structure NaCl-primitive --outdir out/single/NaCl-primitive/
python calculate_madelung_single.py -p 6 --structure CsCl --outdir out/single/CsCl/
python calculate_madelung_single.py -p 6 --structure ZnS-zincblende --outdir out/single/ZnS-zincblende/
python calculate_madelung_single.py -p 6 --structure ZnS-wurtzite --outdir out/single/ZnS-wurtzite/
python calculate_madelung_single.py -p 6 --structure CaF2 --outdir out/single/CaF2/
