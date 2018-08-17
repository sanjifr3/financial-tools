import numpy as np
from tools import financialAnalysisTools as fa
import matplotlib.pyplot as plt

############################ Portfolio ###################################

Portfolio = {
  'BTC': 1,
  'ETH': 1,
  'LTC': 1,
  'XLM': 1,
  'VEN': 1,
  'XRP': 1
}

############################ //Portfolio// ###################################

if __name__ == '__main__':

  total_val = 0
  print 'CryptoCurrency Portfolio'
  for coin, val in Portfolio.items():
    valCAD = fa.getPriceInCAD(val,coin)
    total_val += valCAD

    print "  {c}: ${vc} CAD ({v} {c})".format(c=coin, vc=valCAD, v=val) 

  print "\nTotal: ${} CAD".format(round(total_val,2)) 

  # Price History
  for coin in Portfolio.keys():
    fa.getPriceHistory(coin)

  plt.show()