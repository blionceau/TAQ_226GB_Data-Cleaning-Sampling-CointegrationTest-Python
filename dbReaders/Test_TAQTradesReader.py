import unittest
from dbReaders.TAQTradesReader import TAQTradesReader

class Test_TAQTradesReader(unittest.TestCase):

    def test1(self):
        fileName = 'D:/TAQClean/trades/20070620/IBM_trades.binRT'
        reader = TAQTradesReader( fileName )
        print( reader.getN() )

        print( reader.getTimestamp(1) )

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test1']
    unittest.main()