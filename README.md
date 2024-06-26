# DualBandit
This repository holds the public code for the SMJ paper SMJ-21-23122

This scripts are provided AS-IS. The scripts include some guidance in the code, and the authors of the paper will be happy to provide guidance via email. Please note that some coding experience is required to understand these scripts.

Each of the bandit scripts require a "tasklist" to run. The format of the tasklist is in the sample csv file in this repository, and it has 3 parameters. First parameter is the exploration propensity parameter, the second parameter is the exploration breadth parameter, and the third parameter is currently unused.

I ran these codes as they are on the Ohio Supercomputer, but a personal computer with linux would work just fine. The command line to start the script would look like

$ python Softmax-Softmax (Evaluative).py 100000

The above command would run the script with 100000 firms. You may change the number of firms to fit your needs.

==============

The file alphabeta-second-deriv.py contains a simple python script to verify the mechanism we propose in the paper. The script takes the raw output for a single-parameter bandit, fits a polynomial function of 3rd degree on the data, and, for each turbulence level, calculates the second derivative with respect to knowledge and exploration cost at the optimal performance value. Our mechanism posits that the second derivative with respect to knowledge will be zero or negative, whereas the second derivative with respect to exploration cost will be positive in that point.
