import numpy as np
import pandas as pd
from time import sleep

df = pd.read_csv('4.1.ProcessedLoansWithDaysFilled.csv')
prices = pd.read_csv('5.historicalPrices.csv')

def check_liquidation(row, prices, liq_threshold):
    block_num = min(prices['BlockNo'], key=lambda x: abs(x - row['block_opened']))

    collateral_names = ['weth_collateral_balance','usdc_collateral_balance','wbtc_collateral_balance','comp_collateral_balance','uni_collateral_balance','link_collateral_balance']
    borrow_names = ['weth_borrow_balance','usdc_borrow_balance','wbtc_borrow_balance','comp_borrow_balance','uni_borrow_balance','link_borrow_balance']

    prices_check_counts = 0

    while prices_check_counts <= row['position_days']*2:

        collateral_balance = 0
        borrow_balance = 0
        max_ratio = 0
        
        price_row = prices.loc[prices['BlockNo'] == block_num]
        
        for collateral in collateral_names:
            if collateral == 'weth_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['Eth']*row[collateral]
            if collateral == 'usdc_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['Usd']*row[collateral]
            if collateral == 'wbtc_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['WBtc']*row[collateral]
            if collateral == 'comp_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['Comp']*row[collateral]
            if collateral == 'uni_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['Uni']*row[collateral]
            if collateral == 'link_collateral_balance' and row[collateral] > 0:
                collateral_balance += price_row['Link']*row[collateral]

        for borrow in borrow_names:
            if borrow == 'weth_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['Eth']*row[borrow]
            if borrow == 'usdc_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['Usd']*row[borrow]
            if borrow == 'wbtc_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['WBtc']*row[borrow]
            if borrow == 'comp_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['Comp']*row[borrow]
            if borrow == 'uni_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['Uni']*row[borrow]
            if borrow == 'link_borrow_balance' and row[borrow] > 0:
                borrow_balance += price_row['Link']*row[borrow]

        if float(borrow_balance)/float(collateral_balance) > 0.9:
            print(float(borrow_balance)/float(collateral_balance))
        
        block_num += 3500
        prices_check_counts += 1

liq_threshold = 0.9

for idx, row in df.iterrows():
    print(row['dt_opened'])
    if row['collateral_usd_balance'] != 0 and row['borrow_usd_balance'] != 0:
        check_liquidation(row, prices, liq_threshold)
