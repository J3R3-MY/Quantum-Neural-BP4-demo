### Notes 
- The GB codes are decoded using 32 iterations to better compare with some references from the paper
- The toric codes are decoded using 25 iterations
- For everything else, the naming pattern should be obvious.

- To check if another set of hyperparameters improves ensemble performance it is easiest to train another decoder 
  in the python script, print a log line, and then run the binary again. Obiviously it needs the same name as the old one.

- Each ensemble consists of the following:
    - "baseline-noopt"
    - "hamming-one"
    - "hamming-two"
