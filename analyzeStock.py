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
import csv
import argparse as ap
import traceback
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
  parser.add_argument('-u','--update',help='Update today records?',default=False,required=False)
  args = parser.parse_args()
  verbose = args.verbose
  save = args.save
  update = args.update
  
  #save = False
  
  ## Get Current Date ##
  now = datetime.datetime.now()
  date = now.strftime("%Y-%m-%d")

  df = pd.DataFrame()
  if not new_file:
    df = pd.read_csv('stockAnalysis.csv',sep=',')
    #df = df.set_index('stock')

  stocks = []
  indx = args.ticker.find('.csv')

  if indx != -1:
    stocks = open(args.ticker,'r').read().split('\n')
  else:
    stocks.append(args.ticker)

  for stock in stocks:
    stock = stock.upper()

    if indx != -1 and not new_file and not update and len(df[df['stock'] == stock]) >= 1 and df[df['stock'] == stock]['date'].values[0] == date:
        continue
    
    print 'Analyzing {}'.format(stock)

    try:
      gen_data = fa.analyzeStock(stock,verbose,save)
      gen_data['name'] = gen_data['name'].replace('amp;','')

      newDF = pd.DataFrame.from_dict(gen_data, orient='index')
      newDF = pd.DataFrame(columns=newDF.index.values, data=newDF[0].to_frame().T)

      if newDF['stock'] is None:
        continue

      if new_file:
        df = newDF
        new_file = False
      else:
        if len(df[df['stock'] == stock]) == 1:
          for column in df.columns.values:
            df.loc[df['stock'] == stock,column] = newDF[newDF['stock'] == stock][column]
        else:
          df = pd.concat([df,newDF],axis=0)

      if save:
		df.to_csv('stockAnalysis2.csv',sep=',',index=False)

    except Exception as e:
      print '  Failed to analyze {}'.format(stock)
      print '----Traceback Error-----'
      print traceback.print_exc()
      print '----Traceback Error-----'
