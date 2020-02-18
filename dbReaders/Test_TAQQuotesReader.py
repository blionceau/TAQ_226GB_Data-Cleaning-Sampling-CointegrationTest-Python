import unittest
from dbReaders.TAQQuotesReader import TAQQuotesReader

class Test_TAQQuotesReader(unittest.TestCase):

    def test1(self):
        # This is where I keep my test directory. Please
        # change this location for your own test.
        baseDir = "C:\\Users\\lenovo\\Desktop\\Algo Trading\\HW\\HW1\\PlotOutput\\"
        # This is the file I will open and read for this test
        fileName = '20070620\\MSFT_quotes.binRQ'

        #baseDir = 'I:\\R\\quotes\\'

        reader = TAQQuotesReader( baseDir + fileName )
        
        # Using previously tested readers, test for expected values


        # Header records
        # Number of records
        #self.assertEquals( reader.getN(), 68489 )
        # Seconds from Epoc
        #self.assertEquals( reader.getSecsFromEpocToMidn(), 1182312000 )
        
        # End records
        #self.assertEquals( reader.getMillisFromMidn( reader.getN() - 1 ), 57599000 )
        #self.assertEquals( reader.getBidSize( reader.getN() - 1 ), 20 )
        #self.assertEquals( reader.getAskSize( reader.getN() - 1 ), 16 )
        #self.assertAlmostEquals( reader.getAskPrice( reader.getN() - 1 ), 106.03, 3 )
        #self.assertAlmostEquals( reader.getBidPrice( reader.getN() - 1 ), 106, 3 )
        print(reader.getSecsFromEpocToMidn())
        print(reader.getAskPrice(5))
        print(reader.getBidPrice(5))
        print(reader.getN())
        print(reader.getAskSize(5))
        print()
        '''
            The following is taken from test R-based file readers for
            comparison:
            
            Start of file:
            Header
            > zz$h
            $s
            [1] 1182312000
            $n
            [1] 68489
             
            > colnames(zz$r)
            [1] "m"  "bs" "bp" "as" "ap"
            
            Records
             > zz$r[1,"m"]
                   m 
            34241000 
            > zz$r[1,"bs"]
            bs 
             4 
            > zz$r[1,"bp"]
                bp 
            106.42 
            > zz$r[1,"as"]
             as 
            252 
            > zz$r[1,"ap"]
               ap 
            106.5 
            >
            
            End of file:
            > zz$r[,"m"][zz$h$n]
            [1] 57599000
            > zz$r[,"bs"][zz$h$n]
            [1] 20
            > zz$r[,"bp"][zz$h$n]
            [1] 106
            > zz$r[,"ap"][zz$h$n]
            [1] 106.03
            > zz$r[,"as"][zz$h$n]
            [1] 16
            > 
            
            
            1182312000
            30.450000762939453
            30.450000762939453
            138927
            11
            
            30.450000762939453
            20
            30.450000762939453
            20
            30.450000762939453
            20
            
        '''
        for i in range(450, 700):
            print(reader.getAskPrice(i))
            print(reader.getBidPrice(i))
        print(reader.getBidSize(1))
        print(reader.getAskPrice(2))
        print(reader.getBidSize(2))
        print(reader.getAskPrice(3))
        print(reader.getBidSize(3))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
