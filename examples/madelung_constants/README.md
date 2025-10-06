Example demonstrating calculation of Madelung constants for a number of crystal structures with msmJAX.

Two different varieties of calculation are included.
The first one (`single`) computes the Madelung constant one structure at a time, with MSM parameters chosen for each structure individually.
The second one (`batch`) demonstrates how to set up a single evaluation function working for multiple structures, and vectorizes the calculation over structures.

Sample outputs, including plots, can be found in `example_results`.

The top-level entry point when you want to re-run this example are the [`runscript_single_all_structures.sh`](runscript_single_all_structures.sh) and [`runscript_batch.sh`](runscript_batch.sh) wrappers.
They call the Python scripts [`calculate_madelung_single.py`](calculate_madelung_single.py) and [`calculate_madelung_batch.py`](calculate_madelung_batch.py), which perform the actual calculation, with appropriate parameters.
