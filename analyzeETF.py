#!/bin/usr/python2.7

import urllib2
import pandas as pd
import datetime
from bs4 import BeautifulSoup, Tag
import ast
import numpy as np
import json
from ctypes import *
import matplotlib.pyplot as plt
from csv import reader
import argparse as ap
import sys
import datetime
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

from tools import financialAnalysisTools as fa
from tools import genStats as gs

new_file = False

if __name__ == '__main__':
 
 ## Parse Command Line ##
  parser = ap.ArgumentParser()
  parser.add_argument('-t','--ticker',help='Enter stock ticker symbol or csv file with stock names',required=True)
  parser.add_argument('-v','--verbose',help='Verbose?',default=False,required=False)
  parser.add_argument('-s','--save',help='Save?',default=True,required=False)
  args = parser.parse_args()
  verbose = args.verbose
  save = args.save

  ## Get Current Date ##
  now = datetime.datetime.now()
  date = now.strftime("%Y-%m-%d")   

  df = pd.DataFrame()
  if not new_file:
   df = pd.read_csv('etfAnalysis.csv',sep=',')

  stocks = []
  indx = args.ticker.find('.csv')
  if indx != -1:
    stocks = open(args.ticker,'r').read().split('\n')

  else:
    stocks.append(args.ticker)

  for stock in stocks:
    stock = stock.upper()
    print '\nAnalyzing {}'.format(stock)
    try:
      gen_data = fa.analyzeETF(stock,verbose,save)

      if new_file:
        results = []
        results.append((stock,date,0,gen_data['totalAssets'],gen_data['y3Beta'],gen_data['yd']*100,gen_data['ytdReturn']*100,gen_data['y3Return']*100,gen_data['y5Return']*100))
        df = pd.DataFrame(columns=['stock','date','price','assets','3yBeta','yield','YTD','3yReturn','5yReturn'],data=results)
        new_file = False
      else:
        if len(df[df['stock'] == stock]) == 1 and df[df['stock'] == stock]['date'].values[0] == date:
          df[df['stock'] == stock] = [stock,date,0,gen_data['totalAssets'],gen_data['y3Beta'],gen_data['yd']*100,gen_data['ytdReturn']*100,gen_data['y3Return']*100,gen_data['y5Return']*100]
        else:
          df.loc[stock] = [stock,date,0,gen_data['totalAssets'],gen_data['y3Beta'],gen_data['yd'],gen_data['ytdReturn'],gen_data['y3Return'],gen_data['y5Return']]

      df.to_csv('etfAnalysis.csv',sep=',',index=False)
      
    except:
      print '  Failed to analyze {}'.format(stock)