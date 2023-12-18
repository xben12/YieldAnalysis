# function for full range LP
import numpy as np
import pandas as pd
from datetime import datetime
import lib_const

# Get impermanent loss, qty_change_token0, and token1. 
# code also works with  array input
def get_impermanent_loss_without_range(price_change, is_change_token0=True, b_return_df = False, ret_imp_loss_only = False ):
    #input value check ignored here
    if(is_change_token0):
        price_change_token0 = np.array(price_change)
    else:
        price_change_token0 = 1/(1+np.array(price_change))
    new_price0_sqrt = np.sqrt(1+price_change_token0)
    new_x = 1/new_price0_sqrt
    new_y = new_price0_sqrt
    new_pf_value = (1+price_change_token0)*new_x + new_y
    buyhold_value = (1+price_change_token0) + 1
    imp_loss = new_pf_value/buyhold_value -1
    qty_chg_token0 = new_x -1
    qty_chg_token1 = new_y -1

    if ret_imp_loss_only :
        return imp_loss


    if np.isscalar(price_change):
        result_matrix = np.array([ imp_loss, qty_chg_token0, qty_chg_token1])
    else:
        result_matrix = np.column_stack(( imp_loss, qty_chg_token0, qty_chg_token1))
    
    if (b_return_df):
        # Column names
        column_names = [ 'imp_loss', 'qty_chg_token0', 'qty_chg_token1']
        
        if result_matrix.ndim == 1:
            result_matrix = result_matrix.reshape(1, -1)
        
        # Convert NumPy array to DataFrame with column names
        result_matrix = pd.DataFrame(result_matrix, columns=column_names)    

    return result_matrix


def get_impermanent_loss_given_range(price_change, price_range_down):
    #input value check ignored here
    price_range_up = get_opposite_bin_limit_with_same_liquidity(price_range_down)

    #print(price_change,price_range_down, price_range_up )

    x_0 = 1-1/np.sqrt(1+price_range_up)
    y_0 = 1-np.sqrt(1+price_range_down)

    x_n_raw = 1/np.sqrt(1+price_change)-1/np.sqrt(1+price_range_up)
    y_n_raw = np.sqrt(1+price_change) - np.sqrt(1+price_range_down)

    x_max = 1/np.sqrt(1+price_range_down)-1/np.sqrt(1+price_range_up)
    y_max = np.sqrt(1+price_range_up) - np.sqrt(1+price_range_down)

    # x_n cannot be less than 0, or higher when all y is swapped
    x_n = np.minimum(x_max, np.maximum(x_n_raw, 0))
    y_n = np.minimum(y_max, np.maximum(y_n_raw, 0))

    portfolio_0 = x_0*(1+price_change) + y_0
    portfolio_1 = x_n*(1+price_change) + y_n

    imp_loss = portfolio_1/portfolio_0-1
    return imp_loss





def get_opposite_bin_limit_with_same_liquidity(price_change):
    return -price_change/(1+price_change)

def get_liquidity_boost_given_range(prince_range, benchmark = -0.5):
    if(benchmark > 0):
        benchmark = get_opposite_bin_limit_with_same_liquidity(benchmark)

    boost  = (np.sqrt(1+benchmark) -1 )  / (np.sqrt(1+prince_range) -1 )
    return boost



def get_ETHBTC_poolyield_daily(date_begin_yyyymmdd = "2009-01-01", date_end_yyyymmdd = "3000-01-01"):
    pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed'
    data_file_name = lib_const.get_pool_filename(pool_address)
    
    df = pd.read_csv(data_file_name)

    date_begin = datetime.strptime(date_begin_yyyymmdd, '%Y-%m-%d')
    date_end = datetime.strptime(date_end_yyyymmdd, '%Y-%m-%d')


    # Convert 'date' column to YYYYMMDD format
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.sort_values(by='date',ascending=False)

    df = df[(df['date'] >= date_begin) & (df['date'] <= date_end)]

    df['daily_fee_rate'] = df['feesUSD'] / df['tvlUSD']
    return np.average(df['daily_fee_rate'])

def get_pool_performance_statistic(pool_address, token0, token1, fee_bps, year = -1):

    data_file_name = lib_const.get_pool_filename(pool_address)
    # data_file_name = 'output/output_' + pool_address + '.csv'
    df = pd.read_csv(data_file_name)

    # Convert 'date' column to YYYYMMDD format
    df['date_int'] = df['date']
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.sort_values(by='date',ascending=False)   
    
    #df['date'] = pd.to_datetime(df['date'], unit='s').dt.strftime('%Y%m%d')


    df['year'] = df['date'].dt.strftime('%Y')
    df['YYYYMM'] = df['date'].dt.strftime('%Y%m')
    
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
    df_pool_imp_loss_stats = get_impermanent_loss_without_range(price_change_array, b_return_df=True)


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



# Check if the script is being run as the main program
if __name__ == "__main__":
    # result_matrix = get_impermanent_loss(0.1)
    # print(result_matrix)
    # result_matrix[1]
    # result_matrix = get_impermanent_loss(-1.96991/100) # ,  b_return_df=True)
    # print(result_matrix)
    # result_matrix = get_impermanent_loss([0.1, 0.5, 1])
    # print(result_matrix)
    # result_matrix = get_impermanent_loss([0.1, 0.5, 1], b_return_df=True)
    # print(result_matrix)
    # print("test with range: -----")

    price_change = 0.05
    loss = get_impermanent_loss_given_range(price_change, price_range_down=-0.16)
    print("imp loss",price_change, loss)

    
    price_change = -0.08
    loss = get_impermanent_loss_given_range(price_change, price_range_down=0.16)
    print("imp loss, all y gone",price_change, loss)

    your_range = -0.05
    benchmark = -0.15
    print('boost:',your_range,benchmark, "by", get_liquidity_boost_given_range (your_range,benchmark))

    your_range =np.array([-0.05, -0.1]) 
    benchmark = -0.15
    print('boost:',your_range,benchmark, "by", get_liquidity_boost_given_range (your_range,benchmark))
    
    
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
    print("pool daily yield 2023:", get_ETHBTC_poolyield_daily(date_begin_yyyymmdd = "2023-01-01"))

    
