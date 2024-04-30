import requests
import json
import numpy as np
import pandas as pd
from time import sleep

def get_price(address, block):
    usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    usdt = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    
    f_params = {
        "query":'{ pools(where: {token0: "%s", token1: "%s"}, block: {number: %s} orderBy: liquidity, orderDirection: desc){ id token0Price token1Price liquidity } }'%(address, usdc, str(block)),
        "operationName": None,
        "variables": None,
        }

    s_params = {
        "query":'{ pools(where: {token0: "%s", token1: "%s"}, block: {number: %s} orderBy: liquidity, orderDirection: desc){ id token0Price token1Price liquidity } }'%(address, usdt, str(block)),
        "operationName": None,
        "variables": None,
        }
    headers = {'Content-type': "application/json"}
    url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'


    try:
        f_price = float(requests.post(url, headers=headers, data=json.dumps(f_params)).json()['data']['pools'][0]['token1Price'])
        s_price = float(requests.post(url, headers=headers, data=json.dumps(s_params)).json()['data']['pools'][0]['token1Price'])

        price = (f_price+s_price)/2
    except Exception as e:
        sleep(0.01)
        try:
           price = float(requests.post(url, headers=headers, data=json.dumps(f_params)).json()['data']['pools'][0]['token1Price'])
        except Exception as exc:
            sleep(0.01)
            price = float(requests.post(url, headers=headers, data=json.dumps(s_params)).json()['data']['pools'][0]['token1Price'])

    return price

def volatility(address, block):
    #get 1days timestamp
    k = 0
    one_day_price = np.array([])
    
    for i in range(23):
        block = block - k
        one_day_price = np.append(one_day_price, get_price(address, block))
        k = k+721

    k = 0
    one_week_price = np.array([])

    for i in range(6):
        block = block - k
        one_week_price = np.append(one_week_price, get_price(address, block))
        k = k+17280

    one_day_vol = np.std(one_day_price.reshape(-1,1))
    one_week_vol = np.std(one_week_price.reshape(-1,1))

    return one_day_vol*100, one_week_vol*100

