import genStats as gs
import numpy as np
import pandas as pd

import urllib2
from bs4 import BeautifulSoup, Tag
import json
from ctypes import *
import ast
import datetime
import matplotlib.pyplot as plt
from csv import reader
import time
import sys
from lxml import html
import requests
from exceptions import ValueError
from time import sleep
from collections import OrderedDict

if sys.version_info < (3,):
    from cStringIO import StringIO
else:
    from io import StringIO
    xrange = range
from tokenize import generate_tokens

from forex_python.converter import CurrencyRates
from coinmarketcap import Market

import cookielib
from io import StringIO
from csv import DictReader

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

verbose = False

######################### Web Scrapping #########################

# Relevant Key Terms from Company Financial Sheets
relevant = {'bs12':["Cash and cash equivalents",'Receivables','Inventories','Total current assets',
                    'Intangible assets','Accounts payable','Short-term debt','Total current liabilities',
                    'Long-term debt','Other long-term liabilities','Total liabilities',
                    'Total assets','Other intangible assets',"Total stockholders' equity"],
            'bs3': ["Cash and cash equivalents",'Receivables','Inventories','Total current assets',
                    'Intangible assets','Accounts payable','Short-term debt','Total current liabilities',
                    'Long-term debt','Other long-term liabilities','Total liabilities',
                    'Total assets','Other intangible assets',"Total stockholders' equity"],
            'is12':['Revenue','Gross profit','Net income','Diluted'],
            'is3': ['Revenue','Gross profit','Net income','Diluted'],
            'kr12':['Shares Mil','Tax Rate %','Current Ratio',
                    'Operating Cash Flow USD Mil','Free Cash Flow USD Mil',
                    #'Revenue USD Mil','Revenue CAD Mil',
                    'Operating Cash Flow CAD Mil','Free Cash Flow CAD Mil'],
            'kr3': ['Shares Mil','Tax Rate %','Current Ratio',
                    'Operating Cash Flow USD Mil','Free Cash Flow USD Mil',
                    #'Revenue USD Mil','Revenue CAD Mil',
                    'Operating Cash Flow CAD Mil','Free Cash Flow CAD Mil'],
            'cf3' :['Net cash provided by operating activities',
                    'Net cash used for investing activities',
                    'Net cash provided by (used for) financing activities'],
            'cf12':['Net cash provided by operating activities',
                    'Net cash used for investing activities',
                    'Net cash provided by (used for) financing activities']        
}

# Split line into list on commas while ignoring commas within quotations
def split(a):
    """Split a python-tokenizable expression on comma operators"""
    compos = [-1] # compos stores the positions of the relevant commas in the argument string
    compos.extend(t[2][1] for t in generate_tokens(StringIO(a).readline) if t[1] == ',')
    compos.append(len(a))
    return [ a[compos[i]+1:compos[i+1]] for i in xrange(len(compos)-1)]
  
# Get type of currency
def getCurrency (str):
  if 'USD' in str:
    return 'USD'
  elif 'CAD' in str:
    return 'CAD'
  else:
    return 'OTHER'

def getYahooETFData(ticker):
  url = "http://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
  if verbose: print "Parsing %s"%(url)

  page_acquired = False
  time_out = False
  start = time.time()
  while not page_acquired and (time.time() - start) < 3:
      response = requests.get(url)
      #sleep(2)
      parser = html.fromstring(response.text)
      summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
      summary_data = OrderedDict()
      other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
      summary_json_response = requests.get(other_details_json_link)
      json_loaded_summary =  json.loads(summary_json_response.text)
      page_acquired = 'quoteSummary' in json_loaded_summary

  try:
    json_loaded_summary =  json.loads(summary_json_response.text)
    results = json_loaded_summary['quoteSummary']['result'][0]['defaultKeyStatistics']    
    y5Return = np.float(results['fiveYearAverageReturn']['raw'])
    y3Return = np.float(results['threeYearAverageReturn']['raw'])
    y3Beta = np.float(results['beta3Year']['raw'])
    totalAssets = np.float(results['totalAssets']['raw']) / (10**6)
    yd = np.float(results['yield']['raw'])
    ytdReturn = np.float(results['ytdReturn']['raw'])
    
    results_dict = {
      'y5Return': y5Return,
      'y3Return': y3Return,
      'totalAssets': totalAssets,
      'yd': yd,
      'ytdReturn': ytdReturn,
      'y3Beta':y3Beta
    }
    
    return results_dict
  
  except ValueError:
    print "Failed to parse json response"
    return {"error":"Failed to parse json response"}
    
# Scrap financial information from Yahoo Finance    
def getYahooFinancialInfo(ticker):
  url = "http://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
  if verbose: print "Parsing %s"%(url)
  #sleep(2)

  page_acquired = False
  start = time.time()
  while not page_acquired and (time.time() - start) < 3:
      response = requests.get(url)
      #sleep(2)
      parser = html.fromstring(response.text)
      summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
      summary_data = OrderedDict()
      other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
      summary_json_response = requests.get(other_details_json_link)
      json_loaded_summary =  json.loads(summary_json_response.text)
      page_acquired = 'quoteSummary' in json_loaded_summary

  try:
    json_loaded_summary =  json.loads(summary_json_response.text)
    results = json_loaded_summary['quoteSummary']['result'][0]
      
    #print json.dumps(results, indent=4, sort_keys=True)
    
    y_Target_Est = json_loaded_summary["quoteSummary"]["result"][0]["financialData"]["targetMeanPrice"]['raw']
    earnings_list = json_loaded_summary["quoteSummary"]["result"][0]["calendarEvents"]['earnings']
    eps = json_loaded_summary["quoteSummary"]["result"][0]["defaultKeyStatistics"]["trailingEps"]['raw']
    
    datelist = []
    for i in earnings_list['earningsDate']:
      datelist.append(i['fmt'])
    earnings_date = ' to '.join(datelist)
    for table_data in summary_table:
      raw_table_key = table_data.xpath('.//td[contains(@class,"C(black)")]//text()')
      raw_table_value = table_data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
      table_key = ''.join(raw_table_key).strip()
      table_value = ''.join(raw_table_value).strip()
      summary_data.update({table_key:table_value})
    summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps,'Earnings Date':earnings_date,'ticker':ticker,'url':url})
    
    try:
      div = summary_data['Forward Dividend & Yield'].split(' ')[0]
    except:
      div = 0
    try:
      yd =  summary_data['Forward Dividend & Yield'].split(' (')[1].strip('%)')
    except:
      yd = 0
    
    marketCap = summary_data['Market Cap']
    if 'B' in marketCap:
      marketCap = marketCap.strip('B')
      marketCap = np.float(marketCap) * 1000
    elif 'M' in marketCap:
      marketCap = marketCap.strip('M')
      marketCap = np.float(marketCap)
    sector = results['summaryProfile']['sector'].strip(' ')
    industry = results['summaryProfile']['industry'].strip(' ')
    price = results['financialData']['currentPrice']['raw']
    currency = results['financialData']['financialCurrency']
    beta = 1
    try:
      beta = results['defaultKeyStatistics']['beta']['raw']
    except:
      beta = 1
    eps = results['defaultKeyStatistics']['trailingEps']['raw']
    try:
      peg = results['defaultKeyStatistics']['pegRatio']['raw']
    except:
      peg = 0
    try:
      recommendation = results['financialData']['recommendationMean']['raw']
    except:
      recommendation = 0
    try:
      cr = results['financialData']['currentRatio']['raw']
    except:
      cr = 0
    try:
      revenueGrowth = results['financialData']['revenueGrowth']['raw']
    except:
      revenueGrowth = 0
    
    if div == 'N/A':
      div = 0
    if yd == 'N/A':
      yd = 0
    
    results_dict = {
      'marketCap': marketCap,
      'sector': sector,
      'industry': industry,
      'price': price,
      'currency': currency,
      'beta': beta,
      'EPS': eps,
      'PEG': peg,
      'recommendation': recommendation,
      'CR': cr,
      'revenueGrowth': revenueGrowth,
      'dividend': div,
      'yield': yd
    }

    return results_dict
    
  except ValueError:
    print "Failed to parse json response"
    return {"error":"Failed to parse json response"}

def parse_html_table(table):
  n_columns = 0
  n_rows=0
  column_names = []

  # Find number of rows and columns
  # we also find the column titles if we can
  for row in table.find_all('tr'):
    
    # Determine the number of rows in the table
    td_tags = row.find_all('td')
    if len(td_tags) > 0:
      n_rows+=1
      if n_columns == 0:
        # Set the number of columns for our table
        n_columns = len(td_tags)
        
    # Handle column names if we find them
    th_tags = row.find_all('th') 
    if len(th_tags) > 0 and len(column_names) == 0:
      for th in th_tags:
        column_names.append(th.get_text())
        
  if 'Invalid Date' in column_names:
    return pd.DataFrame(columns = column_names)

  # Safeguard on Column Titles
  if len(column_names) > 0 and len(column_names) != n_columns:
    raise Exception("Column titles do not match the number of columns")

  columns = column_names if len(column_names) > 0 else range(0,n_columns)
  df = pd.DataFrame(columns = columns,
            index= range(0,n_rows))
  row_marker = 0
  for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
      df.iat[row_marker,column_marker] = column.get_text()
      column_marker += 1
    if len(columns) > 0:
      row_marker += 1
      
  # Convert to float if possible
  for col in df:
    try:
      df[col] = df[col].astype(float)
    except ValueError:
      pass
  
  return df    

def parse_url(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'lxml')
  return [(table['id'],parse_html_table(table))\
      for table in soup.find_all('table')]
      
def parse(ticker):
  url = "http://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
  response = requests.get(url)
  if verbose: print "Parsing %s"%(url)
  sleep(4)
  parser = html.fromstring(response.text)
  summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
  summary_data = OrderedDict()
  other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
  summary_json_response = requests.get(other_details_json_link)
  try:
    json_loaded_summary =  json.loads(summary_json_response.text)
    y_Target_Est = json_loaded_summary["quoteSummary"]["result"][0]["financialData"]["targetMeanPrice"]['raw']
    earnings_list = json_loaded_summary["quoteSummary"]["result"][0]["calendarEvents"]['earnings']
    eps = json_loaded_summary["quoteSummary"]["result"][0]["defaultKeyStatistics"]["trailingEps"]['raw']
    datelist = []
    for i in earnings_list['earningsDate']:
      datelist.append(i['fmt'])
    earnings_date = ' to '.join(datelist)
    for table_data in summary_table:
      raw_table_key = table_data.xpath('.//td[contains(@class,"C(black)")]//text()')
      raw_table_value = table_data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
      table_key = ''.join(raw_table_key).strip()
      table_value = ''.join(raw_table_value).strip()
      summary_data.update({table_key:table_value})
    summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps,'Earnings Date':earnings_date,'ticker':ticker,'url':url})
    return summary_data
  except ValueError:
    print "Failed to parse json response"
    return {"error":"Failed to parse json response"}
    
def getPrice(stock):
  url = 'https://ca.finance.yahoo.com/quote/{s}?p={s}'.format(s=stock)
  
  ## Doesn't work ##
    
# Get Long-term growth rate from Yahoo Finance #
def getLTGrowthRate(stock):
  url = 'https://ca.finance.yahoo.com/quote/{s}/analysts?p={s}'.format(s=stock)  
  if verbose: print 'Parsing', url
  headers={'User-Agent':user_agent} 
    
  req = urllib2.Request(url,None,headers) #The assembled request
  response = urllib2.urlopen(req)
  content = response.read()
  
  soup = BeautifulSoup(content, 'html.parser')
  LTGrowthRate = 0.0
  for table in soup.find_all('table'):
    df = parse_html_table(table)
    
    if df.columns.values[0] == 'Growth Estimates':
      stk = df[df['Growth Estimates'] == 'Next 5 Years (per annum)'][stock].values[0].replace('%','')
      ind = df[df['Growth Estimates'] == 'Next 5 Years (per annum)']['Industry'].values[0].replace('%','')
      sec = df[df['Growth Estimates'] == 'Next 5 Years (per annum)']['Sector'].values[0].replace('%','')
      sp500 = df[df['Growth Estimates'] == 'Next 5 Years (per annum)']['S&P 500'].values[0].replace('%','')
      
      LTGrowthRate = np.float(sp500)
      if stk != 'N/A':
        LTGrowthRate = min(LTGrowthRate,np.float(stk)/100)
      if ind != 'N/A':
        LTGrowthRate = min(LTGrowthRate,np.float(ind))
      if sec != 'N/A':
        LTGrowthRate = min(LTGrowthRate,np.float(sec))
        
  return LTGrowthRate
      
# Get financial information from GuruFocus
def getEffIntRateonDebt(t):
  t_sp = t.split('.')
  stk = t
  if len(t_sp) > 1:
    if t_sp[1] == 'TO':
      stk = "TSX:" + t_sp[0]

  url = "https://www.gurufocus.com/term/EffectiveInterestRate/"\
        + stk + "/Effective%252BInterest%252BRate%252Bon%252BDebt/"
        
  if verbose: print "Parsing " + url        
         
  headers={'User-Agent':user_agent} 

  req = urllib2.Request(url,None,headers) #The assembled request
  response = urllib2.urlopen(req)
  content = response.read()
  
  soup = BeautifulSoup(content, 'html.parser')
  
  desc = soup.findAll(attrs={'name':'description'})
  desc = desc[0]['content'].encode('utf-8')
  
  indx = desc.find('Effective Interest Rate on Debt %:')
  if indx is not -1:
    indx += len('Effective Interest Rate on Debt %:') + 1
    int_rate = float(desc[indx:indx+4])/100.0
    if int_rate > 0.01:
      return int_rate
  if verbose: print "Effective interest rate on debt couldn't be determined, using 5%"
  return 0.05

# Get financial info from MorningStar
def getMorningStarFinancialInfo(t, currency, reportType = 'bs', period = 'a', columnYear = 5, number = 3, order = 'asc'):
  '''
  t: stock ticker,
  reportType: is: Income Statement, bs: balance sheet, cf: cash flow, 'kr': key_ratios
  period: 3: Quarterly, 12: Annual
  columnYear: '5' or '10'
  number = units of response data: 1 = None, 2 = Thousands, 3 = Millions, 4 = Billions
  order = asc or desc
  '''
  # Type casting
  if period == 'q':
    period = '3'
  else:
    period = '12'
  period = str(period)
  columnYear = str(columnYear)
  number = str(number)
  type = reportType + period
  
  t_sp = t.split('.')
  
  region = "usa"
  if len(t_sp) > 1:
    if t_sp[1] == 'TO' or t_sp[1] == 'V':
      region = "can"
      t_sp2 = t_sp[0].split('-')
      if len(t_sp2) > 1:
        t_sp[0] = t_sp2[0] + '.' + t_sp2[1]
  
  url = ''  
  if reportType != 'kr': 
    url_str = "http://financials.morningstar.com/ajax/ReportProcess4CSV.html?"\
          + "region=" + region + "&"\
          + "t=" + t_sp[0] + "&"\
          + "reportType=" + reportType + "&"\
          + "period=" + period + "&"\
          + "dataType=A&"\
          + "order=" + order + "&"\
          + "columnYear=" + columnYear + "&"\
          + "number=" + number
  else:
    url_str = "http://financials.morningstar.com/ajax/exportKR2CSV.html?"\
          + "region=" + region + "&"\
          + "t=" + t_sp[0]
  if verbose:  print "Parsing " + url_str

  page_acquired = False
  start = time.time()
  while not page_acquired and (time.time() - start) < 4:
    url = urllib2.urlopen(url_str)
    content = url.read()
    soup = BeautifulSoup(content,'lxml')
    page_acquired = hasattr(soup.body,'p')

  name = ''
  currency = ''
  results = []
  dates = []
  ctr = 0
  stop = False
  
  for line in soup.body.p.prettify().split('\n'):
    ctr += 1
    # print line
    if ctr == 2 and reportType != 'kr' and reportType != 'cf':
      name = line.split("(" + t + ")")[0].strip(' ')
    elif ctr == 3 and reportType != 'kr':
      #currency = getCurrency(line)
      dates = line.split(',')
      dates[0] = 'type'
    elif ctr == 4 and reportType == 'kr':
      dates = line.split(',')
      dates[0] = 'type'
    elif stop:
      break
    else:
      split_line = split(line)
      split_line = [x.strip('"') for x in split_line]
      if len(split_line) and split_line[0] in relevant[type]:
        split_line[0] = split_line[0].replace(' USD','').replace(' CAD','').replace(' Mil','')
        results.append(split_line)
      if 'Diluted' in split_line:
        stop = True
        
  df = pd.DataFrame(results, columns=dates)
  df['file'] = reportType

  return name, df

######################### CryptoCurrency #########################

c = CurrencyRates()
m = Market()

def getPriceInCAD(price,currency):
  conversion_rate = 1
  if currency == 'USD':
    conversion_rate = np.float(c.get_rate('USD','CAD'))
  elif currency == 'BTC':
    conversion_rate = np.float(m.ticker('bitcoin',convert='CAD')[0]['price_cad'])
  elif currency == 'ETH':
    conversion_rate = np.float(m.ticker('ethereum',convert='CAD')[0]['price_cad'])
  elif currency == 'XRP':
    conversion_rate = np.float(m.ticker('ripple',convert='CAD')[0]['price_cad'])
  elif currency == 'XLM':
    conversion_rate = np.float(m.ticker('stellar',convert='CAD')[0]['price_cad'])
  elif currency == 'LTC':
    conversion_rate = np.float(m.ticker('litecoin',convert='CAD')[0]['price_cad'])
  elif currency == 'VEN':
    conversion_rate = np.float(m.ticker('vechain',convert='CAD')[0]['price_cad'])
  elif currency == 'CNY':
    conversion_rate = np.float(c.get_rate('CNY','CAD'))
  else:
    print 'Unknown currency type!:', currency
  
  return round(price*conversion_rate,2)

def getPriceHistory(y='ETH',x='USDT', show=True):
  # Open url and read it
  url = urllib2.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair="
              + x + "_" + y +
              "&start=1435699200&end=9999999999&period=14400")
  content = url.read()
    
  # Read the url content using Beautiful Soup
  soup = BeautifulSoup(content,'lxml')
  url_dict = ast.literal_eval(soup.body.string)

  # Split the information into a pandas dataframe
  try:
    url_df = pd.DataFrame(url_dict)
  except ValueError:
    print 'Could not grab price history for ', y
    return None
  url_df['date'] = url_df['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
  url_df['date_only'] = url_df['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S").date())

  # Get aggregate price information per day
  df = url_df.groupby('date_only')[['high','low']].mean()

  # Make a date column
  df.reset_index(inplace=True) # Reset the index
  df.rename(columns={'date_only': 'date'}, inplace=True) # Rename date only column

  # Remove dates out of this range [2015-07-30:2016-07-27]
  # df = df[(df['date'] >= date(2015,7,8)) & (df['date'] <= date(2016,7,27))]

  # Compute average price
  df['average'] = 0.5*(df['high'] + df['low']) # Find the average of the high and low
  
  if show:
    df.plot(x='date',y='average',color='blue')
    plt.title(y)
    plt.ylabel(x)
  return df
  
def getMarketPrice(ticker):
  url = "http://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
  # print "Parsing %s"%(url)
  #sleep(2)

  page_acquired = False
  while not page_acquired:
      response = requests.get(url)
      #sleep(2)
      parser = html.fromstring(response.text)
      summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
      summary_data = OrderedDict()
      other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
      summary_json_response = requests.get(other_details_json_link)
      json_loaded_summary =  json.loads(summary_json_response.text)
      page_acquired = 'quoteSummary' in json_loaded_summary

  try:
    json_loaded_summary =  json.loads(summary_json_response.text)
    results = json_loaded_summary['quoteSummary']['result'][0]
    price = results['financialData']['currentPrice']['raw']
    currency = results['financialData']['financialCurrency']
	
    if currency != 'CAD':
		  price = getPriceInCAD(price,currency)
	
    return price, currency
    
  except ValueError:
    print "Failed to parse json response"
    return {"error":"Failed to parse json response"}
  
############################ Evaluate ############################

def getValue(df, row, column):
  value = 0
  try:
    value = df.loc[row][column].replace(',','')
    value = np.float(value)
  except:
    value = 0
    
  return value

def getValues (df, row, columns):

  values = np.zeros(len(columns))
  idx = 0
  
  for column in columns:
    values[idx] = getValue(df,row,column)
    idx += 1

  return values
    
### Criterion #1: Market Capitalization ###
# Note: Invest in either mid cap or large cap companies
def getCap (marketCap):
  type = 'large'
  bad = 0
  if marketCap < 2000:
    type = 'small'
  elif marketCap < 10000:
    type = 'mid'
  
  if type == 'small':
    if verbose: print "      Bad: This a small cap stock with a market cap of {}!".format(marketCap)
    bad = 1
    
  return type, bad
  
### Criterion #2: Stock Market Sector ###
# Note 1: Invest in cyclical sector stocks during a boom market to maximize returns 
# Note 2: Invest in defensive sector stocks during a fair market to minimize risks
def checkSector(sectors,industry):
  bad = 0
  for sector in sectors:
    if sector in gs.defensive and gs.market == 'boom':
      if verbose: print "      Okay: Investing in defensive sector during a boom market!"
      bad = 0.5
    elif sector in gs.cyclical and gs.market == 'fair':
      if verbose: print "      Bad: Investing in cyclical sector during a fair market!"
      bad = 1
  return bad

### Criterion #3: Net Income ###
# Note #1: Invest in companies that have been consistently growing for the last 3 years
# Note #2: Invest in companies that have grown their net income by at least 20% in the last year
def checkNI(netIncome):
  bad = 0
  NIChange = np.zeros(3)
  
  for i in reversed(range(1,4)):
    if netIncome[i] < netIncome[i-1]:
      if verbose: print "      Bad: The net income hasn't been consistently growing for the last 3 years!"
      if verbose: print "        It's net income dropped from {} to {}, {} years ago.".format(netIncome[i-1],netIncome[i],4-i)
      bad = 1
    NIChange[i-1] = (netIncome[i] - netIncome[i-1]) / np.float(netIncome[i-1])
  
  if NIChange[-1] < 0.2:
    if verbose: print "      Bad: It has not grown its income by 20% in the last year!"
    if verbose: print "        It's last year growth rate was {}%.".format(round(NIChange[-1]*100,2))
    bad += 1
  
  return NIChange, bad

### Criterion #4: Sales & Revenue ###
# Note #1: Only invest in businesses whose sales & revenue were consistently growing in the last 3 years!
# Note #2: Only invest in businesses whose sales & revenue were growing at the same rate as their net income!
def checkRevenueProfit(revenue,profit,NIChange):
  RChange = np.zeros(3)
  PChange = np.zeros(3)
  rnbad = 0
  pnbad = 0
  rbad = 0
  pbad = 0
  for i in reversed(range(1,4)):
    if revenue[i] < revenue[i-1]:
      if verbose: print "      Bad: Its revenue hasn't been consistently growing for the last 3 years!"
      if verbose: print "        It's revenue dropped from {} to {}, {} years ago.".format(revenue[i-1],revenue[i],4-i)
      rbad = 1
    if profit[i] < profit[i-1]:
      if verbose: print "      Bad: Its profit hasn't been consistently growing for the last 3 years!"
      if verbose: print "        It's profit dropped from {} to {}, {} years ago.".format(profit[i-1],profit[i],4-i)
      pbad = 1
    
    RChange[i-1] = (revenue[i] - revenue[i-1]) / np.float(revenue[i-1])
    PChange[i-1] = (profit[i] - profit[i-1]) / np.float(profit[i-1])
    
    if np.abs(RChange[i-1] - NIChange[i-1]) > gs.tol:
      if verbose: print "      Bad: Its change in net income is not equivalent to its change in revenue"
      if verbose: print "       It's net income changed by {}%, and its revenue changed by {}%".format(round(NIChange[i-1]*100,2),round(RChange[i-1]*100,2))
      rnbad = 1
    if np.abs(PChange[i-1] - NIChange[i-1]) > gs.tol:
      if verbose: print "      Bad: Its change in net income is not equivalent to its change in profit"
      if verbose: print "        It's net income changed by {}%, and its profit changed by {}%".format(round(NIChange[i-1]*100,2),round(PChange[i-1]*100,2))
      pnbad = 1
  return RChange, PChange, rnbad + pnbad + pbad + rbad
  
### Criterion #5: Operating Cash Flow ###
# Note #1: Only invest in companies with stable or increasing cash flows!
# Note #2: Avoid investing in businesses with negative operating cash flows
def checkOCF(operatingCF):
  bad = 0
  OCFChange = np.zeros(3)
  for i in reversed(range(1,4)):
    if operatingCF[i] < operatingCF[i-1]:
      if verbose: print "      Bad: Its operating cash flow is decreasing!"
      if verbose: print "        It's operating cash flow dropped from {} to {}, {} years ago.".format(operatingCF[i-1], operatingCF[i], 4-i)
      bad = 1
    OCFChange[i-1] = (operatingCF[i] - operatingCF[i-1]) / np.float(operatingCF[i-1])

  if operatingCF[-1] < 0:
    if verbose: print "      Bad: It has a negative operating cash flow of {}!".format(operatingCF[-1])
    bad += 1
  
  return OCFChange, bad
  
### Criterion #6: Earnings Per Share ###  
# Note #1: Invest in companies whose income statement shows the current quarter's EPS grew over 20%
#   compared with the EPS reported on the same quarter as the previous year!
# Note #2: Only invest in businesses who have a long-term EPS growth rate higher than 10%
def checkEPS(EPSbyQuarter,EPSbyYear):
  EPSGrowthRate = (EPSbyQuarter[-1] - EPSbyQuarter[0]) / np.float(EPSbyQuarter[0])
  LTEPSGrowthRate = np.zeros(3)
  bad = 0
  
  if EPSGrowthRate < 0.2:
    if verbose: print "      Bad: It has not grown its EPS by 20% in the last year!"
    if verbose: print "        It's last year EPS growth rate was {}%.".format(round(EPSGrowthRate*100,2))
    bad += 1

  for i in reversed(range(1,4)):
    LTEPSGrowthRate[i-1] = (EPSbyYear[i] - EPSbyYear[i-1]) / np.float(EPSbyYear[i-1])

  LTEPSGrowthRate = LTEPSGrowthRate.mean()
  
  if LTEPSGrowthRate < 0.1:
    if verbose: print "      Bad: Its long-term EPS growth rate is less than 10%"
    if verbose: print "        It's long term EPS growth rate is {}%".format(round(LTEPSGrowthRate*100,2))
    bad += 1

  return EPSGrowthRate, LTEPSGrowthRate, bad

### Criterion #7: Return on Equity ###
# Note #1: Only invest in companies who have an average ROE over the last 3-5 years greater than 12%
# Note #2: Only invest in companies whose previous year's ROE was greater than 15%
def checkROE(netIncome, shareholdersEq):
  ROE = np.zeros(4)
  bad = 0
  for i in range(0,4):
    ROE[i] = netIncome[i] / np.float(shareholdersEq[i])
  
  if ROE.mean() < 0.12:
    if verbose: print "      Bad: Its average ROE for the last 4 years is less than 12% and is {}%".format(round(ROE.mean()*100,2))
    bad += 1
  
  if ROE[-1] < 0.15:
    if verbose: print "      Bad: Its last year ROE was less than 15% and is at {}%".format(round(ROE[-1]*100,2))
    bad += 1
  
  return ROE, bad

### Criterion #8: Debt Settlement Capacity ###
# Note #1: Only invest in companies whose Free Cash Flow x 3 > Long-term Debt
# Note 2: For companies with high short-term debt, only invest in them if their current ratio is greater than
#  1.5. Definitely do not put your money into company whose current ratio is less than 1  
# Current Ratio

def checkCR(freeCF, LTDebt, currentAssets, currentLiabilities, CRy):
  bad = 0
  if freeCF*3 < LTDebt:
    if verbose: print "      Bad: It doesn't have enough free cash flow to settle its long-term debts in 3 years"
    if verbose: print "        3x its free cash flow is ${} million, and its long-term debt is ${} million".format(freeCF*3, LTDebt) 
    bad += 1
  
  CR = 0
  try:
    CR = currentAssets/np.float(currentLiabilities)
  except:
    CR = CRy
  
  if CR < 1:
    if verbose: print "      Bad: Its current ratio is less than 1 and is {}!".format(CR)
    bad += 1
  elif CR < 1.5:
    if verbose: print "      Ok: Its current short-term debt is {} million and its current CR is {}".format(currentLiabilities, CR)
    bad += 0.5
  
  return CR, bad

### Criterion #9: PEG Ratio ###
# Price to Earning (P/E) Ratio = Market Value per Share / Earnings per Share
# A low PE could indicate the company is undervalued. A high PE could indicate the company is overvalued.
# Note #1: Only buy stocks whose PEG ratio is less than to equal to 1!
  # Price/Earning to Growth (PEG) Ratio:
def checkPEG(price, EPS, PEG):
  PE = price/EPS
  if PEG > 1:
    if verbose: print "      Bad: It has a PEG ratio of {} which is greater than 1!".format(PEG) 
    return 1
  return 0

### Criterion #10: Intrinsic Value ###
# Note #1: Only buy stocks whose intrinsic value is greater than the current market price!
def getIntrinsicPrice(cash,debt,STDebt,LTDebt,beta,estInterestRate,busTaxRate,marketCap,shares,price,freeCF,LTGrowthRate):
  # Calculate
  debtMarketVal = (STDebt + LTDebt)*1.2 # millions
  GDPGrowthRate = np.mean(gs.GDPGrowthRate) / 100.0
  
  # STEP 2: Calculate Weighted Average Cost of Capital (WACC) as a discount rate
  RE = gs.riskFreeRate + beta*gs.marketRiskPremium # Cost of Equity # Risk Free Rate + Stock Beta x Market Risk Premium (using CAPM)
  RD = estInterestRate*(1-busTaxRate) # Cost of Debt # Interest Rate * (1- Tax Rate)
  WACC = (marketCap/(marketCap + debtMarketVal)*RE) + (debtMarketVal/(marketCap + debtMarketVal)*RD)

  # Intrinsic Value per Share = (Present Value + Terminal Value + Cash - Debt) / (Total Number of Shares Outstanding)

  # STEP 3: Calculate the Discounted Free Cash Flows 
  projectedFCF = np.zeros(10)
  discountFactor = np.zeros(10)
  discountedFCF = np.zeros(10)

  for n in range(1,11):
    projectedFCF[n-1] = freeCF * (1 + LTGrowthRate)**n
    discountFactor[n-1] = 1.0 / (1 + WACC)**n
    discountedFCF[n-1] = projectedFCF[n-1] * discountFactor[n-1]
    
  # STEP 4: Calculate the Present Value of 10-Year Free Cash Flow
  presentVal = discountedFCF.sum() # million

  # Calculate the Perpetuity Value
  perpVal = (projectedFCF[-1]*(1+GDPGrowthRate))/(WACC-GDPGrowthRate)
  terminalVal = perpVal*discountFactor[-1] # million

  # STEP 5: Calculate the Intrinsic Value
  ## Intrinsic Value per share = (Present Value + Terminal Value + Cash - Debt) / Total Number of Shares Outstanding
  intrinsicVal = (presentVal + terminalVal + cash - debt)
  intrinsicValPS = intrinsicVal / shares
  
  pctDiff = intrinsicValPS/price

  valuation = 'CorrectlyValued'
  if pctDiff > 1:
    #print "Undervalued: This stock's market value is {}% of its intrinsic value".format(round(pctDiff,2))
    valuation = 'Undervalued'
  elif pctDiff < 1:
    #print "Overvalued: This stock's market value is {}% of its intrinsic value".format(round(pctDiff,2))
    valuation = 'Overvalued'
  
  return intrinsicValPS, pctDiff, valuation

def checkFCF(financingCF):
  bad = 0
  FCFChange = np.zeros(3)
  for i in reversed(range(1,4)):
    if financingCF[i] > financingCF[i-1]:
      if verbose: print "      Bad: Its financing cash flow is increasing!"
      if verbose: print "        It's financing cash flow increased from {} to {}, {} years ago.".format(financingCF[i-1], financingCF[i], 4-i)
      bad = 0.5
    FCFChange[i-1] = (financingCF[i] - financingCF[i-1]) / np.float(financingCF[i-1])

  if financingCF[-1] > 0:
    if verbose: print "      Bad: It has a positive financing cash flow of {}!".format(financingCF[-1])
    bad += 1
  
  return FCFChange, bad

def checkICF(investingCF):
  bad = 0
  ICFChange = np.zeros(3)
  for i in reversed(range(1,4)):
    if investingCF[i] > investingCF[i-1]:
      if verbose: print "      Bad: Its investing cash flow is increasing!"
      if verbose: print "        It's investing cash flow increased from {} to {}, {} years ago.".format(investingCF[i-1], investingCF[i], 4-i)
      bad = 0.5
    ICFChange[i-1] = (investingCF[i] - investingCF[i-1]) / np.float(investingCF[i-1])

  if investingCF[-1] > 0:
    if verbose: print "      Bad: It has a positive investing cash flow of {}!".format(investingCF[-1])
    bad += 1
  
  return ICFChange, bad  

def getAnnualColumns(DF):
  month = 01
  year = 01

  for val in DF.columns.values:
    indx = val.find('20')
    if indx != -1:
      if int(val[2:4]) > year:
        year = int(val[2:4])
        month = int(val[5:])
      elif int(val[2:4]) == year and int(val[5:]) > month:
        month = int(val[5:])
  
  years_of_interest = []
  for i in reversed(range(0,4)):
    year_str = str(year-i)
    if year-i < 10:
      year_str = '0' + year_str
    month_str = str(month)
    if month < 10:
      month_str = '0' + month_str
    years_of_interest.append('20' + year_str + '-' + month_str)

  return years_of_interest

def getQuarterlyColumns(DF):
  month = 01
  year = 01

  for val in DF.columns.values:
    indx = val.find('20')
    if indx != -1:
      if int(val[2:4]) > year:
        year = int(val[2:4])
        month = int(val[5:])
      elif int(val[2:4]) == year and int(val[5:]) > month:
        month = int(val[5:])
  
  quarters_of_interest = []
  month_str = str(month)
  if month < 10:
    month_str = '0' + month_str

  return ['20' + str(year-1) + '-' + month_str,'20' + str(year) + '-' + month_str]

def getFinancialInfo(stock, verbose = False, save = True):
  ## Get Yahoo Financial Info ##

  ## Get Current Date ##
  now = datetime.datetime.now()
  date = now.strftime("%Y-%m-%d")

  ## Pull information from Yahoo ##
  gen_data = getYahooFinancialInfo(stock)
  gen_data['stock'] = stock
  gen_data['date'] = date

  ## Get GuruFocus Financial Info (Estimated Interest Rate on Debt)##
  estInterestRate = getEffIntRateonDebt(stock)
  
  ## Get Long-term Interest Rate from Yahoo ##
  LTGrowthRate = getLTGrowthRate(stock)  

  ## Get MorningStar Financial Info ##
  gen_data['name'], BSQdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'bs','q')
  _, BSAdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'bs','a')
  _, ISQdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'is','q')
  _, ISAdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'is','a')
  _, KRAdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'kr','a')
  _, KRQdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'kr','q')
  _, CFQdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'cf','q')
  _, CFAdf = getMorningStarFinancialInfo(stock,gen_data['currency'],'cf','a')
  
  ## Merge annual DataFrames ## 
  KRAdf = KRAdf[ISAdf.columns.values] # Drop all old years
  BSAdf = BSAdf.reindex(columns=BSAdf.columns.tolist() + ['TTM']) # Add TTM column to BS annual DF
  ADF = pd.concat([KRAdf,ISAdf,BSAdf,CFAdf]) # Merge annual DataFrames

  ## Merge quarter DataFrames ##
  BSQdf = BSQdf.reindex(columns = BSQdf.columns.tolist() + ['TTM']) # Add TTM column to BS quarterly DF
  QDF = pd.concat([ISQdf,BSQdf,CFQdf])

  ADF = ADF.set_index('type')
  QDF = QDF.set_index('type')

  DF = pd.concat([ADF, QDF.drop(['TTM','file'],axis=1)], axis=1)

  busTaxRate = getValue(ADF,'Tax Rate %','TTM')
  if busTaxRate != '':
    busTaxRate = np.float(busTaxRate)/100.0
  else:
    busTaxRate = 0.4

  # Get year columns
  years =  getAnnualColumns(ADF)
  
  # Get quarter columns
  quarters =  getQuarterlyColumns(QDF)

  gen_data['LTGrowthRate'] = LTGrowthRate
  gen_data['estInterestRate'] = estInterestRate
  gen_data['busTaxRate'] = busTaxRate  
  gen_data['cash'] = getValue(QDF,'Cash and cash equivalents',quarters[-1]) # 'Cash and cash equivalent'
  gen_data['debt'] = getValue(QDF,'Total liabilities',quarters[-1]) # 'Total liabilities'
  gen_data['STDebt'] = getValue(QDF,'Short-term debt',quarters[-1]) # 'Short-term debt'
  gen_data['LTDebt'] = getValue(QDF,'Long-term debt',quarters[-1]) # 'Long-term debt'
  gen_data['LTDebt'] += getValue(QDF,'Other long-term liabilities',quarters[-1]) # 'Other long-term liabilities'
  gen_data['shares'] = getValue(ADF,'Shares','TTM')
  gen_data['freeCF'] = getValue(ADF,'Free Cash Flow','TTM')

  gen_data['intrinsicPrice'] ,gen_data['pctDiff'], gen_data['valuation'] = getIntrinsicPrice(gen_data['cash'],
    gen_data['debt'],gen_data['STDebt'],gen_data['LTDebt'],gen_data['beta'],gen_data['estInterestRate'],
    gen_data['busTaxRate'],gen_data['marketCap'],gen_data['shares'],gen_data['price'],gen_data['freeCF'],
    gen_data['LTGrowthRate'])

  return gen_data, DF, ADF, QDF, years, quarters

def analyzeStock(stock,verbose,save):
  gen_data, DF, ADF, QDF, years, quarters = getFinancialInfo(stock,verbose,save)
  if verbose:
    print "{} ({}) - {}".format(gen_data['name'],gen_data['stock'],gen_data['date'])
    print "  Price: ${} {}".format(gen_data['price'],gen_data['currency'])
    print "  Intrinsic Price: ${} {}".format(round(gen_data['intrinsicPrice'],2),gen_data['currency'])
    print "  Dividend: ${} (Yield: {}%)".format(gen_data['dividend'],gen_data['yield'])
    print "  Valuation: {}".format(gen_data['valuation'])
    print "  Yahoo Recommendation: {}".format(gen_data['recommendation'])
    print "    Problems:"  

  redFlagCtr = 0

  ## Criterion #1:
  gen_data['stockType'],ctr = getCap(gen_data['marketCap'])  
  redFlagCtr += ctr

  # Criterion #2:
  redFlagCtr += checkSector(gen_data['sector'], gen_data['industry'])

  # Criterion #3:
  netIncome = getValues(ADF, 'Net income', years)
  NIChange, ctr = checkNI(netIncome)
  redFlagCtr += ctr

  # Criterion #4:
  revenue = getValues(ADF, 'Revenue', years)
  profit = getValues(ADF,'Gross profit', years)
  RChange, PChange, ctr = checkRevenueProfit(revenue,profit,NIChange)
  redFlagCtr += ctr

  # Criterion #5:
  operatingCF = getValues(ADF,'Operating Cash Flow', years)
  OCFChange, ctr = checkOCF(operatingCF)
  redFlagCtr += ctr

  # Criterion 6
  EPSbyYear = getValues(ADF,'Diluted',years)
  EPSbyQuarter = getValues(QDF, 'Diluted', quarters)
  gen_data['EPSGrowthRate'], gen_data['LongtermEPSGrowthRate'], ctr = checkEPS(EPSbyQuarter,EPSbyYear)
  redFlagCtr += ctr
    
  # Criterion 7
  shareholdersEq = getValues(ADF,"Total stockholders' equity", years)
  ROE, ctr = checkROE(netIncome, shareholdersEq)
  redFlagCtr += ctr

  # Criterion 8
  gen_data['currentAssets'] = getValue(QDF,'Total current assets',quarters[-1]) # 'Total current assets'
  gen_data['currentLiabilities'] = getValue(QDF,'Total current liabilities',quarters[-1]) # 'Total current liabilities'
  CR, ctr = checkCR(gen_data['freeCF'],gen_data['LTDebt'],gen_data['currentAssets'],gen_data['currentLiabilities'],gen_data['CR'])
  redFlagCtr += ctr

  # Criterion 9
  redFlagCtr += checkPEG(gen_data['price'], gen_data['EPS'], gen_data['PEG'])
  
  ## Tele Criteria ##
  
  # Beta < 1, preferably beta < 0.8
  if gen_data['beta'] > 1:
    if verbose: print "      Bad: Beta is greater than 1 so stock is volatile ({})!".format(gen_data['beta'])
    redFlagCtr += 1
  elif gen_data['beta'] > 0.8:
    if verbose: print "      OK: Beta is between 0.8 and 1 so stock is somewhat volatile but less volatile than the market ({})!".format(gen_data['beta'])
    redFlagCtr += 0.5

  ## Check to make sure that investing activities are negative and decreasing
  investingCF = getValues(ADF,"Net cash used for investing activities",years)
  ICFChange, ctr = checkICF(investingCF)
  redFlagCtr += ctr

  ## Check to make sure that financing activities are negative and decreasing
  financingCF = getValues(ADF,"Net cash provided by (used for) financing activities",years)
  FCFChange, ctr = checkFCF(financingCF)
  redFlagCtr += ctr
     
  if verbose: print "    {} has a total of {} red flags!".format(gen_data['stock'],redFlagCtr)
  gen_data['redFlags'] = redFlagCtr

  # Alternative Criteria #

  # Note #1: High inventories is bad
  # Note #2: High accounts receivable is bad
  # Note #3: High accounts payable is bad
  # Note #4: High intangible assets is good

  DF.to_csv('DataSheets/' + gen_data['date'] + '_' + gen_data['stock'] + '_Financials.csv',sep=',')  

  gen_data['AR'] = getValue(QDF,'Receivables',quarters[-1]) # 'Receivable'
  gen_data['inventories'] = getValue(QDF,'Inventories',quarters[-1]) # 'Inventories'
  gen_data['intangibleAssets'] = getValue(QDF,'Intangible assets',quarters[-1]) # 'Intangible Assets'
  gen_data['intangibleAssets'] += getValue(QDF,'Other intangible assets',quarters[-1]) # 'Other intangible assets'
  gen_data['AP'] = getValue(QDF,'Accounts payable',quarters[-1]) # 'Accounts payable'
  
  if gen_data['currentAssets'] != 0:
    gen_data['inventoriesPct'] = gen_data['inventories']/np.float(gen_data['currentAssets'])*100
    gen_data['ARPct'] = gen_data['AR']/np.float(gen_data['currentAssets'])*100
  else:
    if verbose: print "    Note: No current assets so using total assets in denominator"
    totalAssets = getValue(QDF,"Total assets",quarters[-1])
    gen_data['inventoriesPct'] = gen_data['inventories']/totalAssets*100.0
    gen_data['ARPct'] = gen_data['AR']/totalAssets*100.0

  if gen_data['currentLiabilities'] != 0:
    gen_data['APPct'] = gen_data['AP']/np.float(gen_data['currentLiabilities'])*100
  else:
    if verbose: print "    Note: No current assets so using total liabilities in denominator"
    totalLiabilities = getValue(QDF,"Total liabilities",quarters[-1])
    gen_data['APPct'] = gen_data['AP']/totalLiabilities*100

  if verbose:
    print "    Its inventories is equal to {}% of its current assets".format(round(gen_data['inventoriesPct'],2))
    print "    Its accounts receivable is equal to {}% of its current assets".format(round(gen_data['ARPct'],2))
    print "    Its accounts payable is equal to {}% of its current liabilities".format(round(gen_data['APPct'],2))
    print "    Its has intangible assets of ${} million".format(gen_data['intangibleAssets'])  

  if verbose:
    for k,v in gen_data.items():
      print k,v
    print '\n',DF
  
  if save:
    DF.to_csv('DataSheets/' + gen_data['date'] + '_' + gen_data['stock'] + '_Financials.csv',sep=',')
    f = open('DataSheets/' + gen_data['date'] + '_' + gen_data['stock'] + '_General_Info.csv','wb')
    f.write("stockTicker,{}".format(gen_data['stock']))
    f.write('\nname,"{}"'.format(gen_data['name']))
    f.write("\ndate,{}".format(gen_data['date']))
    f.write("\nsector,{}".format(gen_data['sector']))
    f.write("\nindustry,{}".format(gen_data['industry']))
    f.write("\ncurrency,{}".format(gen_data['currency']))
    f.write("\nprice,{}".format(gen_data['price']))
    f.write("\nintrinsicPrice,{}".format(gen_data['intrinsicPrice']))
    f.write("\npctDiff,{}".format(gen_data['pctDiff']))
    f.write("\nvaluation,{}".format(gen_data['valuation']))
    f.write("\nyahooRecommendation,{}".format(gen_data['recommendation']))    
    f.write("\ndividend,{}".format(gen_data['dividend']))
    f.write("\ndividendYield,{}".format(gen_data['yield']))
    f.write("\nbeta,{}".format(gen_data['beta']))
    f.write("\nEPS,{}".format(gen_data['EPS']))
    f.write("\nlongtermEPSGrowthRate,{}".format(gen_data['LongtermEPSGrowthRate']*100))
    f.write("\nEPSGrowthRate,{}".format(gen_data['EPSGrowthRate']*100))
    f.write("\nstockType,{}".format(gen_data['stockType']))
    f.write("\nmarketCap,{}".format(gen_data['marketCap']))
    f.write("\nPEG,{}".format(gen_data['PEG']))
    f.write("\nCR,{}".format(gen_data['CR']))
    f.write("\nrevenueGrowthRate,{}".format(gen_data['revenueGrowth']*100.0))
    f.write("\nlongtermGrowthRate,{}".format(gen_data['LTGrowthRate']*100))
    f.write("\nestimatedInterestRate,{}".format(gen_data['estInterestRate']*100))
    f.write("\nbusinessTaxRate,{}".format(gen_data['busTaxRate']*100))
    f.write("\ncash,{}".format(gen_data['cash']))
    f.write("\ndebt,{}".format(gen_data['debt']))
    f.write("\nshortermDebt,{}".format(gen_data['STDebt']))
    f.write("\nlongtermDebt,{}".format(gen_data['LTDebt']))
    f.write("\nshares,{}".format(gen_data['shares']))
    f.write("\nfreeCashFlow,{}".format(gen_data['freeCF']))
    f.write("\ntotalCurrentAssets,{}".format(gen_data['currentAssets']))
    f.write("\ntotalCurrentLiabilities,{}".format(gen_data['currentLiabilities']))
    f.write("\naccountsReceivable,{}".format(gen_data['AR']))
    f.write("\naccountsReceivablePct,{}".format(gen_data['ARPct']))
    f.write("\naccountsPayable,{}".format(gen_data['AP']))
    f.write("\naccountsPayablePct,{}".format(gen_data['APPct']))
    f.write("\ninventories,{}".format(gen_data['inventories']))
    f.write("\ninventoriesPct,{}".format(gen_data['inventoriesPct']))
    f.write("\nintangibleAssets,{}".format(gen_data['intangibleAssets']))
    f.close()

  return gen_data

### Criterion #11: Future Prospects ###

# Note #1: Identify the business has any strategic plans for the future. 
#   Identify the business plans to release any new product lines.
#   Identify if there are any incoming opportunities for the business to grow.
#  Find out if the management thinks about the future of the business
#  Find out what the analysts think about the potential of the business.

def analyzeETF(stock,verbose,save):

  ## Get Yahoo Financial Info ##
  gen_data = getYahooETFData(stock)

  ## Get Current Date ##
  now = datetime.datetime.now()
  date = now.strftime("%Y-%m-%d")

  
  if verbose:
    print "\n\n{} - {}".format(stock,date)
    for k,v in gen_data.items():
      print k,v

  if save:
    f = open('DataSheets/' + date + '_' + stock + '_ETF_Info.csv','wb')
    f.write("stockTicker,{}".format(stock))
    f.write("\ndate,{}".format(date))
    f.write("\ntotalAssets,{}".format(gen_data['totalAssets']))
    f.write("\n3YearBeta,{}".format(gen_data['y3Beta']))
    f.write("\nyield,{}".format(gen_data['yd']))
    f.write("\nYTDReturn,{}".format(gen_data['ytdReturn']*100))
    f.write("\n3YearReturn,{}".format(gen_data['y3Return']*100))
    f.write("\n5YearReturn,{}".format(gen_data['y5Return']*100))
    f.close()

  return gen_data
