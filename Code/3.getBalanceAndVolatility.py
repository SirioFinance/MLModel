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


dataset = np.array([])

tokens = {
    '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',#UNI
    '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',#WBTC
    '0x514910771af9ca656af840dff83e8264ecf986ca',#LINK
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',#USD
    '0xc00e94cb662c3520282e6f5717214004a7f26888',#COMP
    '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'#WETH
    }

borrows = {
    'weth_borrow_balance':'0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
    'usdc_borrow_balance':'0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'wbtc_borrow_balance':'0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
    'comp_borrow_balance':'0xc00e94cb662c3520282e6f5717214004a7f26888',
    'uni_borrow_balance':'0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
    'link_borrow_balance':'0x514910771af9ca656af840dff83e8264ecf986ca'
    }

collaterals = {
    'weth_collateral_balance':'0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
    'usdc_collateral_balance':'0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'wbtc_collateral_balance':'0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
    'comp_collateral_balance':'0xc00e94cb662c3520282e6f5717214004a7f26888',
    'uni_collateral_balance':'0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
    'link_collateral_balance':'0x514910771af9ca656af840dff83e8264ecf986ca'
    }


df = pd.read_csv('2.RawLoans.csv')
balances = np.array([])

#columns order - collateral_dollar_balance|borrow_dollar_balance|borrow_collateral_ratio|dt_opened|block_opened|position_days|hour_opened|day_opened|month_opened|weth_borrow_balance|usdc_borrow_balance|wbtc_borrow_balance|comp_borrow_balance|uni_borrow_balance|link_borrow_balance|weth_collateral_balance|usdc_collateral_balance|wbtc_collateral_balance|comp_collateral_balance|uni_collateral_balance|link_collateral_balance

for idx, row in df.iterrows():
    block = row['block_opened']

    borrow_balances = np.array([])
    collateral_balances = np.array([])

    borrow_dollar_balance = 0
    collateral_dollar_balance = 0


    #collaterals
    for collateral in collaterals:
        if row[collateral] == 0:
            collateral_balances = np.append(collateral_balances, 0)
            continue
        if row[collateral] != 0:
            usd_amount = get_price(collaterals[collateral], block)*row[collateral]
            collateral_balances = np.append(collateral_balances,usd_amount)
            collateral_dollar_balance = collateral_dollar_balance + usd_amount
            continue

    #borrows
    for borrow in borrows:
        if row[borrow] == 0:
            borrow_balances = np.append(borrow_balances, 0)
            continue
        if row[borrow] != 0:
            usd_balance = get_price(borrows[borrow], block)*row[borrow]
            borrow_balances = np.append(borrow_balances,usd_balance)
            borrow_dollar_balance = borrow_dollar_balance + usd_balance
            continue

    try:
        if borrow_dollar_balance/collateral_dollar_balance < 1:
            balances = np.append(balances, [collateral_dollar_balance, borrow_dollar_balance, borrow_dollar_balance/collateral_dollar_balance])
            balances = np.append(balances, row[1:])
            balances = np.append(balances, borrow_balances/borrow_dollar_balance)
            balances = np.append(balances, collateral_balances/collateral_dollar_balance)

    except:
        try:
            balances = np.append(balances, [collateral_dollar_balance, 0, 0])
            balances = np.append(balances, row[1:])
            balances = np.append(balances, borrow_balances/borrow_dollar_balance)
            balances = np.append(balances, collateral_balances/collateral_dollar_balance)
        except:
            try:
                balances = np.append(balances, [0, borrow_dollar_balance, 0])
                balances = np.append(balances, row[1:])
                balances = np.append(balances, borrow_balances/borrow_dollar_balance)
                balances = np.append(balances, collateral_balances/collateral_dollar_balance)
            except:
                balances = np.append(balances, [0, 0, 0])
                balances = np.append(balances, row[1:])
                balances = np.append(balances, borrow_balances/borrow_dollar_balance)
                balances = np.append(balances, collateral_balances/collateral_dollar_balance)

    print(row['block_opened'])

    
balances = balances.reshape(-1, 33)[:, [3,4,5,6,7,8,9,15,21,27,10,16,22,28,11,17,23,29,12,18,24,30,13,19,25,31,14,20,26,32,0,1,2]]

headerList = ['dt_opened', 'block_opened','position_days', 'hour_opened', 'day_opened', 'month_opened', 'weth_borrow_balance', 'weth_collateral_balance', 'weth_borrow_balance_share', 'weth_collateral_balance_share','usdc_borrow_balance', 'usdc_collateral_balance', 'usdc_borrow_balance_share', 'usdc_collateral_balance_share','wbtc_borrow_balance', 'wbtc_collateral_balance', 'wbtc_borrow_balance_share', 'wbtc_collateral_balance_share','comp_borrow_balance', 'comp_collateral_balance', 'comp_borrow_balance_share', 'comp_collateral_balance_share','uni_borrow_balance', 'uni_collateral_balance', 'uni_borrow_balance_share', 'uni_collateral_balance_share','link_borrow_balance', 'link_collateral_balance', 'link_borrow_balance_share', 'link_collateral_balance_share','collateral_usd_balance', 'borrow_usd_balance', 'borrow-collateral-ratio']     

df = pd.DataFrame(balances, columns = headerList)
df.to_csv("3.ProcessedLoans.csv")
