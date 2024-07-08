# Sirio AI Model Development

## Introduction
This file aims to document in a simple and intuitive way the development of the AI model used by Sirio Finance. This AI model calculates the potential Liquidation Risk associated with each loan and then, just before executing the transaction, displays it to the user, suggesting how to act in case a too high risk is predicted.

Here's a link to learn more about Sirio Finance: https://astrid.gitbook.io/sirio
Here's a link to learn more about liquidation: https://astrid.gitbook.io/sirio/technical/liquidation

The development of the model can be summarized in two steps:
1) Data aggregation, processing, and dataset development
2) Training and testing of various architectures, model development, and deployment in a production environment

In this branch, we address the first step, which will be further detailed in the following phases:
1) Design and implementation of the Missing Data Problem management strategy
2) Borrower Data Gathering
3) Loan Data Processing (Loan Length, Dollar Balances of Loans, Volatility of Assets Used)

There are two folders:
1) Data
2) Code

In the second one there's the enumerated code for each step (extraction, processing). In the first one, there are the enumerated outputs of each step. The first file, in both cases, is a Volatility Analysis. What is it useful for?

## Volatility Analysis
Everyone can notice that there's a lack of Data regarding loans submitted with Hedera Ecosystem Tokens. So, we have to build a Dataset with Tokens that acts similarly to Hedera Tokens. The best thing we could do, is to track volatility of these Tokens and compare them with Tokens listed in Compound Markets, that are the data we are going to use to Train our Model.

In this way, if there are high correlations, we are able to associate a Copound-listed token with Hedera Ecosystem tokens, thanks to their correlation in the volatility behaviours.

The code shows all the steps done to extract data; we downloaded a list of all history daily prices of Hedera and Compound Tokens, then we scaled all of them to the token with minimum number of days price history, and built a correlation Matrix. n the 'Data' folder, there's the correlation Matrix with data required.
