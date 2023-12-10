import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import library_liquiditypool as lib_liq


print("If don't have data downloaded, please run data fetcher first!")

# Load the CSV file
df = pd.read_csv('output/crypto_prices_with_currency.csv')

# Filter rows related to ETH price in terms of BTC
df = df[(df['Token'] == 'ETH') & (df['vs_currency'] == 'btc') & (df['Date'] >= '2021-07-01')]

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])
df['DateYMD'] = df['Date']

# Group by 'Token' and the month of the 'Date' column
grouped = df.groupby(['Token', df['Date'].dt.month])

# Calculate the first date and close price of each month for each group
df['Date_Month_Begin'] = grouped['Date'].transform('min')
df['Close_Month_Begin'] = grouped['Close'].transform('first')
df['Price_chg_vs_MM01'] = df['Close'] / df['Close_Month_Begin'] -1




eth_btc_df = df
# Calculate monthly return (30 day intervals)
eth_btc_df['Date'] = pd.to_datetime(eth_btc_df['Date'])
eth_btc_df.set_index('Date', inplace=True)
eth_btc_df = eth_btc_df[~eth_btc_df.index.duplicated(keep='last')] # Remove duplicates by taking the last value for each date

# Calculate monthly returns
monthly_returns = eth_btc_df['Close'].resample('30D').ffill().pct_change()
monthly_returns.dropna(inplace=True)


# Define benchmark LP range
range_benchm_down =-0.3

# define range, and get coverage and boost x. 
range_down = np.arange(-0.5, 0, 0.02)
range_up = lib_liq.get_price_range_same_liquidity(range_down)

array_price_range = np.column_stack((range_down, range_up))

# now solve for coverage rate: 
df_mon_ret = eth_btc_df['Price_chg_vs_MM01'][eth_btc_df['Date_Month_Begin'] != eth_btc_df['DateYMD']]
ratio_coverage = np.empty(array_price_range.shape[0])
ratio_boost = np.empty(array_price_range.shape[0])

# Count the total number of elements
total_elements = len(df_mon_ret)

for i in range(array_price_range.shape[0]):
    lower_bound = array_price_range[i,0]
    upper_bound = array_price_range[i,1]
    # Count the number of elements within the specified range
    within_range = df_mon_ret.between(lower_bound, upper_bound).sum()

    # Calculate the percentage
    ratio_coverage[i] = (within_range / total_elements)
    ratio_boost[i] = lib_liq.get_liquidity_boost_by_range(lower_bound, benchmark=range_benchm_down)

array_price_range = np.column_stack((range_down, range_up, ratio_coverage, ratio_boost))


import library_pooldata_analyser as pool_ana
pool_daily_yield = pool_ana.get_ETHBTC_poolyield_daily()
# print("pool daily return", pool_daily_yield)


range_perfm = np.empty((array_price_range.shape[0], 3))

for range_i in range(array_price_range.shape[0]):

    indiv_range = np.empty((len(monthly_returns), 3))

    for mon_i in range(len(monthly_returns)):
        mon_ret = monthly_returns.iloc[mon_i]
        range_i_down = array_price_range[range_i, 0 ]
        imp_loss = lib_liq.get_impermanent_loss_range_pos(mon_ret, range_i_down)
        range_cover_ratio = array_price_range[range_i, 2 ]
        boost_ratio =  array_price_range[range_i, 3]
        gross_gain = pool_daily_yield*30*boost_ratio*range_cover_ratio
        gross_loss = imp_loss
        net_gain_loss = gross_gain + gross_loss
        indiv_range[mon_i, :] = np.array([gross_gain, gross_loss, net_gain_loss])

    column_means = np.mean(indiv_range, axis=0)     
    range_perfm[range_i, :] = column_means


result = np.hstack((array_price_range, range_perfm))
print(result)
# Column names
column_names = ['range_limit_down', 'range_limit_up', 'coverage_rate', 'boost_ratio', 'gross_fee_gain', 'imp_loss', 'net_gain']


df_result = pd.DataFrame(result, columns=column_names)

# multiply with 12 convert from monthly to be yearly
columns_to_multiply = ['gross_fee_gain', 'imp_loss', 'net_gain']
df_result[columns_to_multiply] = df_result[columns_to_multiply] * 12


result_file_name = 'output/eth_btc_lp_range_result_v2.csv' 
df_result.to_csv(result_file_name, index=False)
print("result saved to ", result_file_name)


# plot the results


# Columns for x and y
x_column = 'range_limit_down'
y_columns = columns_to_multiply

# Plotting the data
for y_column in y_columns:
    plt.plot(df_result[x_column], df_result[y_column], marker='o', label=y_column)

# Adding labels and a title
plt.xlabel("LP range limit (down part)")
plt.ylabel('value')
plt.title('LP yield from WBTC/ETH pool against range')

# Adding a legend
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# Show the plot
plt.show()

