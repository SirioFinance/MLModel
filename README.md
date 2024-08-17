# Sirio AI Model Development

## Introduction
This document aims to provide a simple and intuitive overview of the development of the AI model used by Sirio Finance. This AI model calculates the potential Liquidation Risk associated with each loan and displays it to the user just before executing the transaction, suggesting actions if a high risk is predicted.

The development of the model can be summarized in three steps:

1) Data aggregation, processing, and dataset development
2) Training and testing of various DL and ML architectures
3) Deployment of the final model in a production environment

This branch focuses on the first step, detailed in the following phases:

1) Design and implementation of the Missing Data Problem management strategy
2) Borrowers Data Gathering
3) Loan Data Processing (Loan Length, Dollar Balances of Loans, Volatility of Assets Used)
4) Final Dataset

There are three main folders:

1) `Data`: Contains the outputs of each step.
2) `Code`: Contains the code for each step (extraction, processing).
3) `dStorage`: Contains the script that manages submissions and retrievals of files from Filecoin decentralized storage.

## Volatility Analysis
There is a lack of data regarding loans with Hedera Ecosystem Tokens since we are among the first Lending & Borrowing Open-Source Protocols on Hedera. Therefore, we need to build a dataset with tokens that act similarly to Hedera tokens. The best approach is to track the volatility of these tokens and compare them with tokens listed in Compound Markets, which are the data we will use to train our model.

By identifying high correlations, we can relate Compound-listed tokens with Hedera Ecosystem tokens through their volatility behavior.

The code demonstrates the steps taken to extract data. We downloaded a list of all historical daily prices of Hedera and Compound Tokens, scaled all of them to the token with the minimum number of days of price history, and built a correlation matrix. The output is stored in the `Data` folder as a Correlation Matrix PNG file.

## Data Extraction
We used GraphQL and Messari architecture to fetch data from Compound Markets, gathering data on loans from block number 15,000,000 (August 2022) to 19,000,000 (January 2024). The resulting dataset is stored in `/Data/2.RawLoans.csv`. For each loan, we gathered:

1)The block number and datetime the loan was started
2)Number of days of duration (if unavailable, we built a Gaussian distribution to cover the missing data gap)
3)Quantity and types of tokens collateralized or borrowed
Data Processing

To evaluate risks, it's essential to define the value in dollars of collateral and borrows. The `/Code/getBalanceAndVolatility.py` script performs this task. For each loan, at the time of transaction submission, it uses SubGraphs to fetch the historical price of the assets at the time of block extraction, enabling the calculation of:

1) Collateral/Borrow Share of each asset in USD (How much of WETH/WBTC/UNI/LINK(COMP is used as collateral/Borrow?)
2) Total Collateral Balance in USD
3) Total Borrow Balance in USD
4) Borrow/Collateral Ratio in USD

## Missing Data Issue & Get Prices Script
The code used to build the Gaussian Distribution model is located in `/Code/4.getDaysExact.py`. The output is the expected value of 259 days, along with a graph representing the constructed Gaussian Distribution.

In this paragraph, we also briefly mention the script located in `/Code/5.getPrices.py`, through which we obtain the historical prices of all assets listed on Compound (USDC, WETH, WBTC, UNI, COMP, LINK) from block number 15,400,000 (around 08-26-2022) to block number 19,999,000 (on 06-01-2024). These prices will be used to determine whether the loans for which we have collected data will be liquidated or not.

## Liquidation Simulation and Final Dataset Output

In the last script, we retrieve all the loan data that was previously extracted and try to determine whether these loans were liquidated. Before explaining the logical approach of our code, let's first walk through the liquidation process:

1. Suppose a user wants to borrow $200 worth of WETH. First, they deposit sufficient collateral, let's say $500 worth of WBTC. At this point, their `Health Factor is calculated as 200 (Dollar value of borrowed assets) / 500 (Dollar value of collateral) = 40%`.
2. Over time, the value of WBTC and WETH changes. Let's assume that after 10 days, the value of WBTC has decreased by 10% and the value of WETH has increased by 20%. Recalculating the `Health Factor: 240 / 450 = 53%`.
3. Suppose that over time, the value of WBTC remains stable, but WETH's value increases exponentially, and the borrowed assets are now worth $420. This means the `Health Factor is now 420 / 450 = 93%`.
4. The Health Factor has exceeded the critical value of 90%, defined as the `Liquidation Threshold`. Beyond this value, the loan becomes liquidatable. This means that a third party can repay the loan (in this case, $420) and receive the borrower's collateral in exchange ($450). The profit is the difference between the two ($30), which is the amount lost by the borrower.

Our goal is to predict the probability of this event occurring. We have data on the loans made (when they started, how many days they lasted, the assets deposited as collateral/borrowed) and, through the previous step (`Prices Script`), we have the prices of all the assets. What do we do? We simulate the loans of which we gathered data, being closed after X days (the amount indicated by the `position_days` value) and observe what value the `Health Factor` assumes during this period. If it exceeds the `Liquidation Threshold` of 90% at any point, the `liquidation_event_happened` value will be switched to `True`. This is what happens in `Code/6.simulateLiquidation.py` and the output is stored in `Data/5.Dataset.csv`, the final version of our Dataset.
