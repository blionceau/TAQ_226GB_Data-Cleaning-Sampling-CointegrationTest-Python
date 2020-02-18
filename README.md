# TAQ_226GB_Data-Cleaning-Sampling-CointegrationTest
TAQ- June -September 2008 NYSE- Data wrangling 

Part A. Preparing the TAQ Data
The TAQ Data preparation:
1. Download the TAQ Data from NYSE. You will ob-
tain the trades and quotes on all stock on the NYSE on business days
throughout the period June 20, 2007 through September 20, 2007. Note
that while you are welcome to work with all the stocks,and concider working with the constituents of the S&P 500 in-
dex.
2. Python method TAQAdjust.py is used to adjust prices, quotes and
number of shares for corporate actions such as stock splits/etc. by using
the “Cumulative Factor to Adjust Prices” and “Cumulative Factor to Ad-
just Shares/Vol” (see, “sp500.xlsx”). Note that if these factors did not
change for a particular stock during the period, then no adjustment is nec-
essary. 2

3. Python method TAQCleaner.py cleans the adjusted TAQ
Data using the procedure described below. Your methods should give you
the option to store the cleaned data to files.

Part B. Compute statistics
1. Python method is created to compute X-minute returns of trades and mid-
quotes, where X is arbitrary.
2. Python class called TAQStats.py calculates basic statistics
of the TAQ Data. Basic statistics to calculate for each stock are:
a. Sample length in days.
b. Total number of trades, total number of quotes, fraction of trades
to quotes.
c. Based on trade and mid-quote returns: mean return (annualized),
median return (annualized), standard deviation of return (annual-
ized), median absolute deviation (annualized), skew, kurtosis, 10
largest returns, 10 smallest returns.

Part C. Cointegration Test

The Granger-Engle cointegration test is applied on a matrix of stock returns, and perform efficient computation and updating for pairs trading by implementing a dynamic approach of using data structures in python and algorithms
for tracking the condition of cointegration in real time.

