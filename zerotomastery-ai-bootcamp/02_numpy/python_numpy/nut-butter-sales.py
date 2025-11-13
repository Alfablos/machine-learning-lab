import numpy as np
import pandas as pd


sales_by_week = np.array([[2, 7, 1],
                          [9, 4, 16],
                          [11, 14, 18],
                          [13, 13, 16],
                          [15,18,9]])

prices = np.array([10, 8, 12])


earnings_by_product = sales_by_week.dot(prices)

butters=["Almond butter", "Peanut butter", "Cashew butter"]

sales_by_week = pd.DataFrame(data=sales_by_week,
                             index=["Mon", "Tue", "Wed", "Thu", "Fri"],
                             columns=butters)

# prices_pd = pd.DataFrame(data=prices.reshape(1, 3),
#                       index=["Price"],
#                       columns=butters)
#
# print(prices_pd)



sales_by_week["Total ($)"] = earnings_by_product

print(sales_by_week)


