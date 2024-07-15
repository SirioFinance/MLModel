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
    url = 'https://gateway-arbitrum.network.thegraph.com/api/f9e80f49ed3fdb526dfbede28eaf2c2d/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'


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

tokens = [
    '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',#UNI
    '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',#WBTC
    '0x514910771af9ca656af840dff83e8264ecf986ca',#LINK
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',#USD
    '0xc00e94cb662c3520282e6f5717214004a7f26888',#COMP
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'#WETH
    ]

prices = np.array([])
block_no = 15400000

while block_no <= 20000000:
    prices = np.append(prices, block_no)
    
    for token in tokens:
        prices = np.append(prices, get_price(token, block_no))
    
    block_no += 3500
    print(block_no)

headerList = ['BlockNo','Uni','WBtc','Link','Usd','Comp','Eth']     

df = pd.DataFrame(prices.reshape(-1, 7), columns = headerList)
df.to_csv("historicalPrices.csv")
