import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import library_logic as lib_liq


print("If you don't have data downloaded, please run data fetcher first!")


def get_df_daily_fees(date_begin_yyyymmdd = "2009-01-01", date_end_yyyymmdd = "3000-01-01"):
    pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed'
    data_file_name = 'output/output_' + pool_address + '.csv'
    df = pd.read_csv(data_file_name)

    # Convert 'date' column to YYYY-MM-DD format
    df['date_int'] = df['date']
    df['date'] = pd.to_datetime(df['date_int'], unit='s').dt.strftime('%Y-%m-%d')
    df['YYYYMM'] = pd.to_datetime(df['date_int'], unit='s').dt.strftime('%Y%m')
    df = df.sort_values(by='date',ascending=False)

    df = df[(df['date'] >= date_begin_yyyymmdd) & (df['date'] <= date_end_yyyymmdd)]

    df['daily_fee_rate'] = df['feesUSD'] / df['tvlUSD']
    df['date_int'] = pd.to_datetime(df['date_int'],  unit='s')
    df.set_index('date_int', inplace=True)
    return df[['date', 'YYYYMM', 'feesUSD', 'tvlUSD', 'daily_fee_rate']]

def get_df_daily_price(data_start_yyyy_mm_dd = '2022-12-01'):

    # Load the CSV file
    df = pd.read_csv('output/crypto_prices_with_currency.csv')
    df = df.drop(columns=['Open', 'High', 'Low']) # the value is not working

    # Filter rows related to ETH price in terms of BTC
    df = df[(df['Token'] == 'ETH') & (df['vs_currency'] == 'btc') & (df['Date'] >= data_start_yyyy_mm_dd)]

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateYMD'] = df['Date']

    # Group by 'Token' and the month of the 'Date' column
    grouped = df.groupby(['Token', df['Date'].dt.year, df['Date'].dt.month])

    # Calculate the first date and close price of each month for each group
    df['Date_Month_Begin'] = grouped['Date'].transform('min')
    df['Date_Month_End'] = grouped['Date'].transform('max')
    df['Close_Month_Begin'] = grouped['Close'].transform('first')
    df['Close_Month_End'] = grouped['Close'].transform('last')
    df['Price_chg_vs_MM01'] = df['Close'] / df['Close_Month_Begin'] -1
    df.drop(df[ df['Date_Month_End'].dt.day < 21 ].index, inplace=True) # filter out months with data < 3 weeks

    eth_btc_df = df
    # Calculate monthly return (30 day intervals)
    eth_btc_df['DateYMD'] = pd.to_datetime(eth_btc_df['DateYMD'])
    eth_btc_df.set_index('DateYMD', inplace=True)
    eth_btc_df = eth_btc_df[~eth_btc_df.index.duplicated(keep='last')] # Remove duplicates by taking the last value for each date
    return eth_btc_df


def get_df_comb_price_fee(df_price, df_fee):
    df = pd.merge(df_price, df_fee, left_index=True, right_index=True)
    return df

def get_mon_performance_by_range(range_down, df, benchmark_down = -0.3):
        df_mon_chg = df[[ 'YYYYMM', 'Price_chg_vs_MM01']][df['Date_Month_End'] == df['Date']]
        df_mon_chg.dropna(inplace=True)

        df_mon_day_ret = df[['YYYYMM',  'Price_chg_vs_MM01','daily_fee_rate' ]][df['Date_Month_Begin'] != df['Date']]

        lower_bound = range_down
        benchmark_lower = benchmark_down
        upper_bound = lib_liq.get_bin_price_range_same_liquidity(lower_bound)

        ret_columns = ['YYYYMM', 'range_down', 'mon_total_price_chg', 'mon_total_fee_yield', 'coverage_rate', 'boost_factor', 'gross_return', 'imp_loss',  'net_return']

        boost_factor = lib_liq.get_liquidity_boost_by_range(prince_range=lower_bound, benchmark=benchmark_lower)
        result_mon = np.empty((len(df_mon_chg), len(ret_columns)))

        for mon_i in range(len(df_mon_chg)):
                yyyymm = df_mon_chg['YYYYMM'].iloc[mon_i]

                mon_total_price_chg = df_mon_chg['Price_chg_vs_MM01'].iloc[mon_i]

                df_yyyymm =  df_mon_day_ret[df_mon_day_ret['YYYYMM'] == yyyymm]
                mon_total_observ = len(df_yyyymm)
                mon_within_range =df_yyyymm['Price_chg_vs_MM01'].between(lower_bound, upper_bound).sum()
                coverage_rate= (mon_within_range / mon_total_observ)

                mon_total_fee_yield = df_yyyymm['daily_fee_rate'].sum()

                gross_return = mon_total_fee_yield*coverage_rate*boost_factor

                imp_loss = lib_liq.get_impermanent_loss_range_pos(mon_total_price_chg, lower_bound)

                net_return = (1+gross_return)*(1+imp_loss) -1

                result_mon[mon_i, :] =np.array([int(yyyymm),lower_bound,mon_total_price_chg, mon_total_fee_yield,coverage_rate, boost_factor,gross_return, imp_loss, net_return ])

        df_mon_result = pd.DataFrame(data=result_mon, columns=ret_columns)
        return df_mon_result

def show_simulation_result(df_result, x_column, y_cols_name, main_y_col_name, y_annualise_factor = 12):
      
    # multiply with 12 convert from monthly to be yearly
    df_result[y_cols_name] = df_result[y_cols_name] * y_annualise_factor

    # Plotting the data
    for y_column in y_cols_name:
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
    plt.show()

    # plt.plot(df_result[x_column], df_result[main_y_col_name], marker='o', label=y_column)
    # plt.show()







def get_full_range_performance(range_down, df, benchmark_range =-0.3):
    array_range_rst =  np.empty((len(range_down),4))
    for range_i in range(len(range_down)):
            range_down_i = range_down[range_i]
            df_i = get_mon_performance_by_range(range_down_i, df, benchmark_down=benchmark_range)
            average_gross_return = df_i['gross_return'].median() #.mean()
            average_imp_loss = df_i['imp_loss'].median() #mean()
            average_net_return = df_i['net_return'].median() #.mean()
            array_range_rst[range_i, :] = np.array([range_down_i, average_gross_return,average_imp_loss,average_net_return ])

    ret_columns = ['range_limit_down', 'gross_fee_gain', 'imp_loss', 'net_gain']
    len(ret_columns)
    df_result = pd.DataFrame(data=array_range_rst, columns=ret_columns)

    return df_result



def main ():
    data_start_yyyy_mm_dd = '2022-12-01'
    range_down = np.arange(-0.5, 0, 0.02)
    benchmark_range = -0.3


    df_price = get_df_daily_price(data_start_yyyy_mm_dd)
    df_fee = get_df_daily_fees(date_begin_yyyymmdd = data_start_yyyy_mm_dd)
    df = get_df_comb_price_fee(df_price, df_fee)
    df_result = get_full_range_performance(range_down, df, benchmark_range=benchmark_range)

    result_file_name = 'output/eth_btc_lp_range_result_v3.csv' 
    df_result.to_csv(result_file_name, index=False)
    print("result saved to ", result_file_name)

    y_cols_name = ['gross_fee_gain', 'imp_loss', 'net_gain']
    x_column = 'range_limit_down'
    main_y_col_name = 'net_gain'

    print("show net_yield, gross_yield, and imp loss chart.")
    print("Close the chart to end program.")
    show_simulation_result(df_result, x_column, y_cols_name,main_y_col_name, y_annualise_factor=12 )
    
    
if __name__ == "__main__" :
    main()