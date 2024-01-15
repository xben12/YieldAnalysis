import numpy as np
import main as m
from math import sqrt
import pandas as pd

def get_lp_evaluation_scenarios():
    
    
    date_begin = '2021-05-05' # price: 0.060538
    date_end = '2023-12-20' # price: 0.060388

    list_scenarios = [
        ['recent 1 year backtest', '2022-12-15', '2023-12-15'],
        ['1 year price revert-back', '2022-05-13', '2023-05-13'],
        ['1 year maximum price-down','2022-10-30',  '2023-10-30' ],
        ['1 year maximum price-up', '2022-06-12', '2023-06-12' ],
        ['history case max price-down','2021-12-09',  '2022-06-19' ],
        ['history case max price-up','2022-06-19',  '2022-09-09' ], 
        ['history case max first-down-then-up','2021-12-09',  '2022-09-09' ]
        
    ]

    list_scenarios_name = ['scenario_name', 'date_begin', 'date_end']

    df_price = m.get_df_daily_price(date_begin,date_end)
    
    df_scenarios = pd.DataFrame(list_scenarios, columns=list_scenarios_name)
    df_scenarios['index'] = df_scenarios.index
    
    # first get the beign_date price
    df_scenarios['date'] = pd.to_datetime(df_scenarios['date_begin']) 
    df_scenarios.set_index('date', inplace=True)
    df_p_tmp = df_price[['price']].rename(columns={'price': 'begin_price'})
    df_scenarios = pd.merge(df_scenarios, df_p_tmp, left_index=True, right_index=True, how='left'  )


    # then get end_date price
    df_scenarios['date'] = pd.to_datetime(df_scenarios['date_end']) 
    df_scenarios.set_index('date', inplace=True)
    df_p_tmp = df_price[['price']].rename(columns={'price': 'end_price'})
    df_scenarios = pd.merge(df_scenarios, df_p_tmp, left_index=True, right_index=True, how='left'  )

    # reset index as 0,1,... to maintan the sequence
    df_scenarios.set_index("index", inplace=True)
    df_scenarios.sort_index(ascending=True, inplace=True)

    return df_scenarios




def get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_down, range_up, b_input_pct = True):
    
    # convert to price
    if b_input_pct: 
        range_down = p0*(1+range_down)
        range_up = p0*(1+range_up)

    if p0 < range_down:
        p0 = range_down
    elif p0>range_up:
        p0 = range_up
    else:
        p0=p0

    if pn < range_down:
        pn = range_down
    elif pn>range_up:
        pn = range_up
    else:
        pn=pn


    p0_sqrt = sqrt(p0)
    pn_sqrt = sqrt(pn)
    
    p_rg_down_sqrt = sqrt(range_down)
    p_rg_up_sqrt = sqrt(range_up)
    
    if p0 == range_down: # only x has
        Lx = qty0 / (1/p0_sqrt - 1/p_rg_up_sqrt) 
        Ly = 0
    elif p0 == range_up: # only y has 
        Lx =0
        Ly = qty1 / (p0_sqrt - p_rg_down_sqrt)
    else:    
        Lx = qty0 / (1/p0_sqrt - 1/p_rg_up_sqrt) 
        Ly = qty1 / (p0_sqrt - p_rg_down_sqrt)


    if pn < p0: # price down, using y as liquidity
        L = Ly
        pn_sqrt = max(pn_sqrt, p_rg_down_sqrt) 
        final_token0_qty = L*(1/pn_sqrt - 1/p0_sqrt) + qty0
        final_token1_qty = L*(pn_sqrt - p_rg_down_sqrt)
    elif pn > p0: # price up, using x as liquidity
        L = Lx
        pn_sqrt = min(pn_sqrt, p_rg_up_sqrt) 
        final_token0_qty = L*(1/pn_sqrt - 1/p_rg_up_sqrt)
        final_token1_qty = L*(pn_sqrt -p0_sqrt) + qty1
    else:
        final_token0_qty, final_token1_qty = qty0, qty1

    return final_token0_qty, final_token1_qty


import pandas as pd
import lib_logic
# break the data into monthly, start from inital postion, and rebalance monthly. 

range_down = -0.1
range_up = -1*range_down/(1+range_down) 



def portfolio_value_no_rebalance(df, range_down, range_up, initial_qty_0_and_1 = None, benchmark_avg_yld_range = -0.3 ):
    
    df.sort_index(ascending=True)

    starting_price = df['price'].iat[0]
    end_price = df['price'].iat[-1]
    
    if initial_qty_0_and_1 is None:
        initial_qty0 = 1
        initial_qty1 = initial_qty0 * starting_price
    else:
        if(len(initial_qty_0_and_1)!=2):
            raise ValueError("Input error: initial_qty_0_and_1 must have 2 elements!")
        initial_qty0 = initial_qty_0_and_1[0]
        initial_qty1 = initial_qty_0_and_1[1]


    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, benchmark_avg_yld_range )

    p0 = starting_price
    pn = end_price
    
    qty0 = initial_qty0
    qty1 = initial_qty1

    end_qty0, end_qty1 = get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_down, range_up, b_input_pct = True)
        


    range_price_up = p0 * (1+range_up)
    range_price_down = p0 * (1+range_down)
    ps_b_within_range = df['price'].apply(lambda x: int(x>=range_price_down and x<=range_price_up ))
        
    df[ 'price_range_up'] = range_price_up
    df[ 'price_range_down'] = range_price_down
    df[ 'b_within_range'] =  ps_b_within_range


    fee_yield =boost_factor* (df['daily_fee_rate']* ps_b_within_range ).sum()

    value_mon_begin = qty0*p0+qty1
    # note the below formula is simplifation. 
    value_mon_end = end_qty0*pn+end_qty1 + (qty0*pn+qty1)*fee_yield   #is it so? 

    array_col_names = ['p0', 'end_p', 'qty0', 'qty1', 'end_qty0', 'end_qty1', 'month_fee_yield', 'value_mon_begin', 'value_mon_end']

    this_mon_array = np.array([p0, pn, qty0, qty1, end_qty0, end_qty1, fee_yield, value_mon_begin,value_mon_end ])
    rst_df = pd.DataFrame(data=[this_mon_array], columns=array_col_names)

    return rst_df 




def portfolio_monthly_rebalance(df, range_down, range_up, initial_qty_0_and_1 = None ):
    
    df.sort_index(ascending=True)

    starting_price = df['price'].iat[0]
    if initial_qty_0_and_1 is None:
        initial_qty0 = 1
        initial_qty1 = initial_qty0 * starting_price
    else:
        if(len(initial_qty_0_and_1)!=2):
            raise ValueError("Input error: initial_qty_0_and_1 must have 2 elements!")
        initial_qty0 = initial_qty_0_and_1[0]
        initial_qty1 = initial_qty_0_and_1[1]

    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, -0.3 )


    labels_ym = df['YYYYMM'].unique()
    array_col_names = ['p0', 'end_p', 'qty0', 'qty1', 'end_qty0', 'end_qty1', 'month_fee_yield', 'value_mon_begin', 'value_mon_end']

    # Create an empty DataFrame with specified column names
    rst_df_ym = pd.DataFrame(index=labels_ym, columns=array_col_names)



    mon_groups = df.groupby('YYYYMM')
    p0 = starting_price
    qty0 = initial_qty0
    qty1 = initial_qty1
    
    for ym, df_ym in mon_groups: # index is the date. 
        pn = df_ym['price'].iat[-1]
        end_qty0, end_qty1 = get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_down, range_up, b_input_pct = True)
        
        # get 3 value, price_range_up, price_range_down, b_within_range, daily_fee_rate
        # note that if we don't do swap, and allocate the leftover capitcal for LP, our deployed liqudity can be very low. 
        # my final return will be: average_daily_yield * boost * coverage
        # daily_yield * Lx/Lx+Ly * 2 * boost * (1, 0)
        range_price_up = p0 * (1+range_up)
        range_price_down = p0 * (1+range_down)
        ps_b_within_range = df_ym['price'].apply(lambda x: int(x>=range_price_down and x<=range_price_up ))
        
        
        df.loc[df_ym.index, 'price_range_up'] = range_price_up
        df.loc[df_ym.index, 'price_range_down'] = range_price_down
        df.loc[df_ym.index, 'b_within_range'] =  ps_b_within_range
        
        
        month_fee_yield =boost_factor* (df_ym['daily_fee_rate']* ps_b_within_range ).sum()
        value_mon_begin = qty0*p0+qty1
        # note the below formula is simplifation. 
        value_mon_end = end_qty0*pn+end_qty1 + (qty0*pn+qty1)*month_fee_yield   #is it so? 

        
        # no need to calculate imp loss for now. we 
        this_mon_array = np.array([p0, pn, qty0, qty1, end_qty0, end_qty1, month_fee_yield, value_mon_begin,value_mon_end ])
        rst_df_ym.loc[ym] = this_mon_array
        
        # update for next round calc
        # this part calc assumes re-balance (since price range has changed, but deposit exactly same amount)
        # it is like i use qty0 for right liquidity, and qty1 for left side liquidity. 
        p0 = pn
        qty0 = end_qty0 + qty0*month_fee_yield
        qty1 = end_qty1 + qty1*month_fee_yield

    rst_names = ['p0', 'qty0', 'qty1', 'end_p', 'end_qty0', 'end_qty1']
    rst_values = [starting_price,initial_qty0,initial_qty1, rst_df_ym['end_p'].iloc[-1], rst_df_ym['end_qty0'].iloc[-1],rst_df_ym['end_qty1'].iloc[-1]  ]

    # Create a dictionary using zip and dictionary comprehension
    rst_dict = {name: value for name, value in zip(rst_names, rst_values)}


    return rst_dict, rst_df_ym    


if __name__ == "__main__":

    b_get_eval_scenarios = True;
    if b_get_eval_scenarios:
        df_scenarios = get_lp_evaluation_scenarios()
        print(df_scenarios)

    run_rebalance = True
    if run_rebalance:
        date_begin = '2022-12-01'

        df_price = m.get_df_daily_price(date_begin)
        df_fee = m.get_df_daily_fees(date_begin = date_begin)
        df = m.get_df_comb_price_fee(df_price, df_fee)
        print("\n check data df first 3 rows:")
        print(df.head(3))
        
        rst_dict, rst_df_ym = portfolio_monthly_rebalance(df, range_down, range_up )

        rst_df_ym.to_clipboard()
        print("\n results: starting position vs ending position: ")
        print(rst_dict)
        print(rst_df_ym)
        
        
        
        print("\n results: no balance: ")
        benchmark_avg_yld_range = -0.3
        range_down = -0.2 # yearly no rebalance
        range_up = -1*range_down/(1+range_down) 

        rst_df_ym = portfolio_value_no_rebalance(df, range_down, range_up, benchmark_avg_yld_range = benchmark_avg_yld_range )
            
        print(rst_df_ym)