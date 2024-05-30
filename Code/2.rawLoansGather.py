import requests
import json
import numpy as np
from datetime import datetime
from time import sleep
import pandas as pd

def getPositions(dataset,assets_name,block):

    params = {
        "query":'{ positions(where: {side: "BORROWER", blockNumberOpened: %s}){ timestampOpened timestampClosed account{ positions{ side balance asset{ name } } } } }'%str(block),
        "operationName": None,
        "variables": None,
        }
    
    url = 'https://api.thegraph.com/subgraphs/name/messari/compound-v3-ethereum'
    headers = {'Content-type': "application/json"}

    while True:
        try:
            borrows = requests.post(url, headers=headers, data=json.dumps(params)).json()['data']
            break
        except Exception as e:
            print(e)
            sleep(2)
            continue

    if len(borrows['positions']) > 0:
        for borrow in borrows['positions']:

            #borrow and collateral positions balances
            weth_borrow_balance = 0
            wbtc_borrow_balance = 0
            uni_borrow_balance = 0
            link_borrow_balance = 0
            comp_borrow_balance = 0
            usdc_borrow_balance = 0

            weth_collateral_balance = 0
            wbtc_collateral_balance = 0
            uni_collateral_balance = 0
            link_collateral_balance = 0
            comp_collateral_balance = 0
            usdc_collateral_balance = 0
            
            #timestampsposition
            dt_open = datetime.fromtimestamp(int(borrow['timestampOpened']))
            hour_opened = dt_open.hour
            day_opened = dt_open.day
            month_opened = dt_open.month
            dt_str_opened = str(dt_open.day)+'-'+ str(dt_open.month)+'-'+ str(dt_open.year)

            try:
                
                dt_opened = datetime.fromtimestamp(int(borrow['timestampOpened']))
                dt_closed = datetime.fromtimestamp(int(borrow['timestampClosed']))

                td = (dt_closed-dt_opened).total_seconds() / 86400
                position_days = int(td)
            except:
                hour_closed = None
                day_closed = None
                month_closed = None
                position_days= None

            

            #fetch account positions
            for position in borrow['account']['positions']:
                if position['side'] == 'BORROWER':
                    if assets_name[position['asset']['name']] == 'WETH':
                        weth_borrow_balance = weth_borrow_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'USDC':
                        usdc_borrow_balance = usdc_borrow_balance + int(position['balance'])/1000000             
                    if assets_name[position['asset']['name']] == 'WBTC':
                        wbtc_borrow_balance = wbtc_borrow_balance + int(position['balance'])/100000000
                    if assets_name[position['asset']['name']] == 'COMP':
                        comp_borrow_balance = comp_borrow_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'UNI':
                        uni_borrow_balance = uni_borrow_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'LINK':
                        link_borrow_balance = link_borrow_balance + int(position['balance'])/1000000000000000000

                if position['side'] == 'COLLATERAL':
                    if assets_name[position['asset']['name']] == 'WETH':
                        weth_collateral_balance = weth_collateral_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'USDC':
                        usdc_collateral_balance = usdc_collateral_balance + int(position['balance'])/1000000             
                    if assets_name[position['asset']['name']] == 'WBTC':
                        wbtc_collateral_balance = wbtc_collateral_balance + int(position['balance'])/100000000
                    if assets_name[position['asset']['name']] == 'COMP':
                        comp_collateral_balance = comp_collateral_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'UNI':
                        uni_collateral_balance = uni_collateral_balance + int(position['balance'])/1000000000000000000
                    if assets_name[position['asset']['name']] == 'LINK':
                        link_collateral_balance = link_collateral_balance + int(position['balance'])/1000000000000000000

            dataset = np.append(dataset, [dt_str_opened, block, position_days, hour_opened, day_opened, month_opened, weth_borrow_balance, usdc_borrow_balance, wbtc_borrow_balance, comp_borrow_balance, uni_borrow_balance, link_borrow_balance, weth_collateral_balance, usdc_collateral_balance, wbtc_collateral_balance, comp_collateral_balance, uni_collateral_balance, link_collateral_balance])
            return dataset
    else:
        return dataset
    
dataset = np.array([])

assets_name = {
    "Wrapped Ether":"WETH",
    "Compound":"COMP",
    "Coinbase Wrapped Staked ETH":"WETH",
    "Rocket Pool ETH":"WETH",
    "USD Coin":"USDC",
    "Wrapped liquid staked Ether 2.0":"WETH",
    "ChainLink Token":"LINK",
    "Wrapped BTC":"WBTC",
    "Uniswap":"UNI"
    }
#19841138
for block in range(15412445, 19841138):
    print(block)
    dataset = getPositions(dataset,assets_name,block)

headerList = ['dt_opened', 'block_opened','position_days', 'hour_opened', 'day_opened', 'month_opened', 'weth_borrow_balance', 'usdc_borrow_balance', 'wbtc_borrow_balance', 'comp_borrow_balance', 'uni_borrow_balance', 'link_borrow_balance', 'weth_collateral_balance', 'usdc_collateral_balance', 'wbtc_collateral_balance', 'comp_collateral_balance', 'uni_collateral_balance', 'link_collateral_balance']     

df = pd.DataFrame(dataset.reshape(-1, 18), columns = headerList)
df.to_csv("sirioRawLoans.csv")
#np.savetxt("sirioRawLoans.csv", dataset.reshape(-1, 17), delimiter=",", fmt="%s")
  


