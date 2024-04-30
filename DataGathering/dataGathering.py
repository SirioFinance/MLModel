from utils import *
import requests
import json
import numpy as np
from datetime import datetime
from time import sleep

def get_borrows(block, dataset):
    accepted_markets = [
        '0xc3d688b66703497daa19211eedff47f25384cdc31f9840a85d5af5bf1d1762f925bdaddc4201f984',#UNI
        '0xc3d688b66703497daa19211eedff47f25384cdc32260fac5e5542a773aa44fbcfedf7c193bc2c599', #WBTC
        '0xc3d688b66703497daa19211eedff47f25384cdc3514910771af9ca656af840dff83e8264ecf986ca', #LINK
        '0xc3d688b66703497daa19211eedff47f25384cdc3a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',#USD
        '0xc3d688b66703497daa19211eedff47f25384cdc3c00e94cb662c3520282e6f5717214004a7f26888', #COMP
        '0xc3d688b66703497daa19211eedff47f25384cdc3c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2' #WETH
        ]

    tokens = [
        '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',#UNI
        '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',#WBTC
        '0x514910771af9ca656af840dff83e8264ecf986ca',#LINK
        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',#USD
        '0xc00e94cb662c3520282e6f5717214004a7f26888',#COMP
        '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'#WETH
        ]

    params = {
        "query":'{ positions(where: {side: "BORROWER", blockNumberOpened: %s}){ hashOpened market{ id } timestampOpened account{ positions{ side hashClosed market{ id } balance timestampClosed asset{ id } } } } }'%str(block),
        "operationName": None,
        "variables": None,
        }
    
    url = 'https://api.thegraph.com/subgraphs/name/messari/compound-v3-ethereum'
    headers = {'Content-type': "application/json"}

    while True:
        try:
            positions = requests.post(url, headers=headers, data=json.dumps(params)).json()['data']['positions']
            break
        except Exception as e:
            print(e)
            sleep(2)
            continue
            
            
    last_borrow = False
    prices = []

    if len(positions)==0:
        return


    if len(positions)>0:
        for address in tokens:
            prices.append(get_price(address, block))
                
    for borrow in positions:
        if borrow['market']['id'] in accepted_markets:
            
            #analyze each borrow of interesting markets
            borrow_hash = borrow['hashOpened']
            borrow_block = block
                        
            hour = datetime.fromtimestamp(int(borrow['timestampOpened'])).hour
            day = datetime.fromtimestamp(int(borrow['timestampOpened'])).day
            month = datetime.fromtimestamp(int(borrow['timestampOpened'])).month
                        
            total_collateral_usd = 0
            total_borrowed_usd = 0
            last_borrow = borrow['account']['positions']

            collateral_shares = np.array([[0],[0],[0],[0],[0],[0]], dtype=object)

            #collateral positions of borrower account
            for position in borrow['account']['positions']:
                
                if position['market']['id'] in accepted_markets and position['side'] == 'COLLATERAL':
                    amount = float(position['balance'])
                    address = position['asset']['id']

                    #calculate prices of collateral at the time of borrow position open
                    if position['market']['id'] == accepted_markets[1]:
                        amount = amount/100000000
                        collateral_usd = prices[1]*amount
                        collateral_shares[1] = collateral_shares[1][0] + collateral_usd
                                    
                    if position['market']['id'] == accepted_markets[3]:
                        collateral_usd = amount/1000000
                        collateral_shares[3] = collateral_shares[3][0] + collateral_usd
                                    
                    if position['market']['id'] != accepted_markets[1] and position['market']['id'] != accepted_markets[3]:
                        amount = amount/1000000000000000000

                        if position['market']['id'] == accepted_markets[0]:
                            collateral_usd = prices[0]*amount
                            collateral_shares[0] = collateral_shares[0][0] + collateral_usd
                        if position['market']['id'] == accepted_markets[2]:
                            collateral_usd = prices[2]*amount
                            collateral_shares[2] = collateral_shares[2][0] + collateral_usd
                        if position['market']['id'] == accepted_markets[4]:
                            collateral_usd = prices[4]*amount
                            collateral_shares[4] = collateral_shares[4][0] + collateral_usd
                        if position['market']['id'] == accepted_markets[5]:
                            collateral_usd = prices[5]*amount
                            collateral_shares[5] = collateral_shares[5][0] + collateral_usd
                                                            
                    total_collateral_usd = total_collateral_usd + collateral_usd
                    if total_collateral_usd == 0:
                        continue

                #fetch borrower positions of the account
                if position['market']['id'] in accepted_markets and position['side'] == 'BORROWER':
                                
                    amount = float(position['balance'])
                    address = position['asset']['id']
                                
                    if position['market']['id'] == accepted_markets[1]:
                        amount = amount/100000000
                        borrowed_usd = prices[1]*amount
                                    
                    if position['market']['id'] == accepted_markets[3]:
                        borrowed_usd = amount/1000000
                                    
                    if position['market']['id'] != accepted_markets[1] and position['market']['id'] != accepted_markets[3]:
                        amount = amount/1000000000000000000
                        if position['market']['id'] == accepted_markets[0]:
                            borrowed_usd = prices[0]*amount
                        if position['market']['id'] == accepted_markets[2]:
                            borrowed_usd = prices[2]*amount
                        if position['market']['id'] == accepted_markets[4]:
                            borrowed_usd = prices[4]*amount
                        if position['market']['id'] == accepted_markets[5]:
                            borrowed_usd = prices[5]*amount
                                
                    total_borrowed_usd = total_borrowed_usd + borrowed_usd
                    if total_borrowed_usd == 0:
                        continue
                                
            try:
                ratio = total_borrowed_usd/total_collateral_usd
            except Exception as e:
                continue
                              
            if ratio > 0.0000000001:
                print(ratio)
                dataset = np.append(dataset, [borrow_hash, borrow_block, hour, day, month, ratio, total_borrowed_usd, total_collateral_usd, collateral_shares[0][0]/total_collateral_usd, collateral_shares[1][0]/total_collateral_usd, collateral_shares[2][0]/total_collateral_usd, collateral_shares[3][0]/total_collateral_usd, collateral_shares[4][0]/total_collateral_usd, collateral_shares[5][0]/total_collateral_usd])
    return dataset
                    
                    
                    
dataset = np.array([])    

for block in range(15412445, 18500000):
    print(block)
    dataset = get_borrows(block, dataset)

np.savetxt("equilibriaDataset.csv", dataset.reshape(-1, 14), delimiter=",", fmt="%s")
# assign header columns
headerList = ['borrow_hash', 'borrow_block', 'hour', 'day', 'month', 'ratio_loan_collateral', 'total_borrowed_usd', 'total_collateral_usd', 'uni_collateral_share','btc_collateral_share', 'link_collateral_share', 'usd_collateral_share', 'compound_collateral_share', 'eth_collateral_share'] 
  
# open CSV file and assign header 
with open("equilibriaDataset.csv", 'w') as file: 
    dw = csv.DictWriter(file, delimiter=',',  fieldnames=headerList) 
    dw.writeheader() 
  
# display csv file 
#fileContent = pd.read_csv("lendingdata.csv") 
