WARNING: this code is very much a work in progress, and has thus far been primarily used by a single author, so bugs may well exist and certainly documentation will be lacking. Comments / suggestions / questions more than welcome: danieljstrouse@gmail.com.

BACKGROUND

This small library implements the generalized information bottleneck (Strouse, Schwab 2016), which includes the information bottleneck (IB) (Tishby, Pereira, Bialek 1999) and the deterministic information bottleneck (DIB) (Strouse, Schwab 2016) as special cases.

The generalized IB operates on an input distribution p(X,Y) and clusters/compresses X into T such that T has/retains maximal information about Y. Its output is a cluster mapping / compressive encoder q(t|x). The cost function the generalized IB minimizes is:

L[q(t|x)] = H(T)-alpha*H(T|X)-beta*I(T;Y)

alpha=0 is the DIB case, alpha=1 is the IB case, and intermediate values interpolate between the two. beta is a tradeoff parameter between compression and informativeness. For more, see the aforementioned papers.

FILES
 
1) IB.py includes the core functionality to run the generalized IB on data. This is the primary file most will be interested in.
2) data_generation.py includes some example functions to generate synthetic data for the IB. This is just to help you get off the ground quickly, but you may very well wish to run the IB on your own data.
3) experimental_control.py includes some example files written to run IB experiments. These are for illustration only, as they still include details particular to the author (e.g. hard-code file paths).

Within IB.py, there are 2 main functions one might wish to call. To run a single fit of the IB with a single alpha / beta / set of other parameters, use IB_single. However, in most cases, one is interested in the results over a range of beta values, and perhaps other parameters as well, and for this, use IB.

IB accepts as input a pandas dataframe, where rows indicate separate experiments (i.e. settings of the fit parameters) and columns indicate fit parameters. This design choice was made to: 1) facilitate experimentation, and 2) so that the input format matched the output format. It is inspired by the "tidy data" paradigm (Wickham 2014). For each combination of fit parameters, the refine_beta function helps to select the appropriate beta to try out, and the output dataframe will include all such runs. (Note that if you want to hand choose your beta, you should use IB_single.)

In the simplest case, one might wish to run just one version of the generalized IB, say the DIB, on a dataset with the default parameters. To do that, one could create a dataframe with a single column, "alpha", and a single row, with the value of alpha in that row equal to 0. That may sound like an overly complicated way to call a single experiment, but its very helpful when running dozens, such as: testing out many versions of the generalized IB (alpha) over different initializations (p0) with different convergence conditions (ctol_abs, ctol_rel, zeroLtol).

See documentation within each file for more details.

TODOS
# add normalization check to entropy and KL
# set max number of steps/time for single beta?
# make work with sparse data
# improve parallelization
# make IB init that perfectly interpolates between wavy and uni?