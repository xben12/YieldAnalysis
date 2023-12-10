import pandas as pd
import numpy as np
import library_liquiditypool

def get_ETHBTC_poolyield_daily(date_begin_yyyymmdd = "20090101", date_end_yyyymmdd = "30000101"):
    pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed'
    data_file_name = 'output/output_' + pool_address + '.csv'
    df = pd.read_csv(data_file_name)

    # Convert 'date' column to YYYYMMDD format
    df['date_int'] = df['date']
    df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y%m%d')
    df = df.sort_values(by='date',ascending=False)

    df = df[(df['date'] >= date_begin_yyyymmdd) & (df['date'] <= date_end_yyyymmdd)]

    df['daily_fee_rate'] = df['feesUSD'] / df['tvlUSD']
    return np.average(df['daily_fee_rate'])

def get_pool_performance_statistic(pool_address, token0, token1, fee_bps, year = -1):

    data_file_name = 'output/output_' + pool_address + '.csv'
    df = pd.read_csv(data_file_name)

    # Convert 'date' column to YYYYMMDD format
    df['date_int'] = df['date']
    df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y%m%d')
    df = df.sort_values(by='date',ascending=False)

    df['year'] = pd.to_datetime(df['date_int'], unit='s').dt.strftime('%Y')
    df['YYYYMM'] = pd.to_datetime(df['date_int'], unit='s').dt.strftime('%Y%m')
    
    # Filter rows where 'year' is not equal to 2023
    if (year != -1):
        df = df[df['year'] == str(year)]

    if(df.shape[0] < 30): # if data less than a month
        print("Less than 30 days of data. Calc stopped!")
        return pd.DataFrame()
    
    df['daily_fee_rate'] = df['feesUSD'] / df['tvlUSD']
    
    # Calculate log price change
    df['log_price_change'] = np.log(df['token0Price']) - np.log(df['token0Price'].shift(1))

    # Drop the first row as it will have NaN for log price change
    df = df.dropna()

    # whole year can get 364 daily change. the correct way should be using last day of prev year, here we ignore it. 
    real_yield_annualise_factor =  364/df.shape[0] 

    # Calculate the total log price change
    total_log_price_change = df['log_price_change'].sum()
    total_pct_price_change = np.exp(total_log_price_change) -1
    total_lp_yield = df['daily_fee_rate'].sum()



    df['accum_fee_rate_7d'] = df['daily_fee_rate'].rolling(window=7).sum()
    df['price_change_7d'] = df['token0Price'].pct_change(7)

    # Calculate the 95th and 5th percentiles of 'price_change_7d'
    df = df.dropna()
    price_change_7d_95th = np.percentile(df['price_change_7d'], 95,method='nearest')
    price_change_7d_5th = np.percentile(df['price_change_7d'], 5,method='nearest')

    # df.to_csv("output/datacheck.csv")

    # Filter rows corresponding to the percentiles
    accum_fee_rate_7d_95th = df[df['price_change_7d'] == price_change_7d_95th]['accum_fee_rate_7d'].iloc[0]
    accum_fee_rate_7d_5th = df[df['price_change_7d'] == price_change_7d_5th]['accum_fee_rate_7d'].iloc[0]


    price_change_array = [total_pct_price_change, price_change_7d_95th, price_change_7d_5th]
    df_pool_imp_loss_stats = library_liquiditypool.get_impermanent_loss(price_change_array, b_return_df=True)


    # Create a 1-row DataFrame
    df_poolkey = pd.DataFrame({'pool_address': [pool_address], 'token0': [token0], 'token1': [token1], 'year': [year], 'fee_bps':[fee_bps]})

    # Create a 3-row DataFrame
    
    data_scenarios = {'scenario': ['yearly_acc', 'weekly_95th', 'weekly_5th'],
                 'scen_price_chg':[total_pct_price_change,price_change_7d_95th, price_change_7d_5th ],
                 'scen_lp_yield':[total_lp_yield,accum_fee_rate_7d_95th, accum_fee_rate_7d_5th ]} 
                 
    df_scenario = pd.DataFrame(data_scenarios)
    df_scenario['scen_lp_annualised'] = df_scenario['scen_lp_yield']*real_yield_annualise_factor

    # Concatenate the DataFrames column-wise
    # Concatenate the DataFrames column-wise and repeat values
    df_poolkey = pd.concat([df_poolkey] * len(data_scenarios), ignore_index=True)

    result_df = pd.concat([df_poolkey, df_scenario, df_pool_imp_loss_stats], axis=1)
    return result_df


if __name__ == "__main__":
    # Load CSV into DataFrame
    data_pools_input = {
        'poolAddress': ['0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', 
                        '0x4585fe77225b41b697c938b018e2ac67ac5a20c0', 
                        '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640', 
                        '0x5777d92f208679db4b9778590fa3cab3ac9e2168', 
                        '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa'],
        'token0': ['WBTC', 'WBTC', 'USDC', 'DAI', 'WSTETH'],  # 
        'token1': ['WETH', 'WETH', 'WETH', 'USDC', 'WETH'], 
        'fee_bps' : [30,5,5,1,1]
    }

    # Create DataFrame
    df_pool_input = pd.DataFrame(data_pools_input)
    df_pool_stats=pd.DataFrame()

    for index in range(len(df_pool_input)) :
        pool_addr = df_pool_input.at[index,'poolAddress'];
        token0 = df_pool_input.at[index,'token0'];
        token1 = df_pool_input.at[index,'token1'];
        fee_bps = df_pool_input.at[index,'fee_bps'];
        df_result = get_pool_performance_statistic(pool_addr, token0, token1,fee_bps, 2022)
        print(df_result)
        df_pool_stats = pd.concat([df_pool_stats, df_result], ignore_index=True)
    
    df_pool_stats.to_csv('output/result_pools.csv', index=False)

    print("pool daily yield", get_ETHBTC_poolyield_daily())
    print("pool daily yield 2023:", get_ETHBTC_poolyield_daily(date_begin_yyyymmdd = "20230101"))

    
