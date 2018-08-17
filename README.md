# financial-tools
Python tools for analyzing stocks, crpyto, and ETFs

------------------------------------------------------------
## Get information on Stocks

To analyze one stock (<stock_name>):

```python analyzeStocks.py -t <stock_name>```

To analyze multiple stocks (<path_to_csv_file>):

```python analyzeStocks.py -t <path_to_csv_file>```

Verbose Option: (print results? Default: False): ```-v```

Save Option: (updated DataSheets? Default: True): ```-s```

Update Option (update info collected today? Default: False): ```-u```

-------------------------------------------------------------
## Get information on ETFs

To analyze one ETF (<etf_name>):

```python analyzeETF.py -t <etf_name>```

To analyze multiple ETFs (<path_to_csv_file>):

```python analyzeETF.py -t <path_to_csv_file>```

Verbose Option: (print results? Default: False): ```-v```

Save Option: (updated DataSheets? Default: True): ```-s```

-------------------------------------------------------------
## Get Market Price on Crypto

Update portfolio in getCryptoMarketPrice.py

Usage: ```python getCryptoMarketPrice.py```

-------------------------------------------------------------
## Get Market Price on Stocks

Update portfolio in getMarketPrice.py

Usage: ```python getMarketPrice.py```
