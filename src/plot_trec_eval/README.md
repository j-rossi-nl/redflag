# DISCLAIMER
We make use of a modified version of plot-trec_eval repository.  
Original work from https://github.com/cvanweelden/plot-trec_eval  


# plot-trec_eval

Plotting scripts for trec_eval output. Useful for reporting on IR system
evaluations.

These require Python (2.6 or 2.7 or 3.5), 
[NumPy](http://numpy.scipy.org/), 
and [matplotlib](http://matplotlib.org/).

## Precision-recall curves

Plot precision-recall curves. These show the performance over all topics
for ranked retrieval systems.

```
usage: Plot precision-recall curves. [-h] [-f OUTPUT] files [files ...]

positional arguments:
  files                 Pass multiple files to plot all the runs in the same
                        plot.

optional arguments:
  -h, --help            show this help message and exit
  -f OUTPUT, --output OUTPUT
                        Save the figure to specified file.
```
### Example

`$ python3 plot_pr_curve.py $(find ./results/*MAP*txt)`

## Per-topic AP or AP difference

Plot AP per topic for 1 run or per-topic difference for 2 runs.

```
usage: Plot AP per topic for 1 run or per-topic difference for 2 runs.
       [-h] [-f OUTPUT] [-s SORT] files [files ...]

positional arguments:
  files                 When passing two files the plotted difference is
                        f1-f2.

optional arguments:
  -h, --help            show this help message and exit
  -f OUTPUT, --output OUTPUT
                        Save the figure to specified file.
  -s SORT, --sort SORT  Sort the topics in descending AP/difference.
```

### Example

`$ python3 plot_topic_ap.py -s true indri.eval okapi.eval`