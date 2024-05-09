# Sirio AI Model Development
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
