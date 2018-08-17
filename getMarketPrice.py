import numpy as np
from tools import financialAnalysisTools as fa
import matplotlib.pyplot as plt

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

############################ //Portfolio// ###################################

if __name__ == '__main__':

	stock_types = {'Value Stocks': valueStocks, 'Dividend Stocks': dividendStocks, 'ETF Stocks': ETFStocks}
	
	total_value = 0
	for stock_type, stocks in stock_types.items():
		type_value = 0
		print "\n"
		for stock, amt in stocks.items():
			try:
				price, currency = fa.getMarketPrice(stock)
				type_value += price*amt
				total_value += price*amt
				print "    {s}: ${v} CAD (${p} {c} ea.)".format(s=stock, v=price*amt, p=price, c=currency)	
			except:
				print "    {s}: Could not acquire price!".format(s=stock)
			
		print "  {s}: ${p} CAD".format(s=stock_type, p=type_value)
	print "Total Value: ${} CAD".format(total_value)