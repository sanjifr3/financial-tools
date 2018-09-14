#!/usr/bin/env python
import numpy as np
from tools import financialAnalysisTools as fa
import matplotlib.pyplot as plt
import pandas as pd

from_file = True
df = None

############################ Portfolio ###################################

valueStocks = {
  'BIDU': 1,
  'AAPL': 1,
  'AMD': 1,
  'PYPL': 1,
  'HD': 1,
}

dividendStocks = {
}

ETFStocks = {
}

if from_file:
  df = pd.read_csv('my_portfolio.csv', sep=',', names=['stocks'])

############################ //Portfolio// ###################################

if __name__ == '__main__':

  if df is None:
    stock_types = {'Value Stocks': valueStocks, 'Dividend Stocks': dividendStocks, 'ETF Stocks': ETFStocks}
    
    total_value = 0
    for stock_type, stocks in stock_types.items():
      type_value = 0
      print ("\n")
      for stock, amt in stocks.items():
        price, currency = fa.getMarketPrice(stock)
        try:
          type_value += price*amt
          total_value += price*amt
          print ("    {s}: ${v} CAD (${p} {c} ea.)".format(s=stock, v=price*amt, p=price, c=currency))
        except:
          print ("    {s}: Could not acquire price!".format(s=stock))
        
      print ("  {s}: ${p} CAD".format(s=stock_type, p=type_value))
    print ("Total Value: ${} CAD".format(total_value))

  else:
    for i, stocks in df.iterrows():
      stk = stocks.values[0]
      price, currency = fa.getMarketPrice(stk, False)

      if price is not None:
        df.at[i,'price'] = price
        df.at[i,'currency'] = currency
    
    print df

    df.to_csv('my_portfolio_updated.csv',sep=',', index=False)

