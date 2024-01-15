import pandas as pd
import numpy as np
import main as m
from math import sqrt

import lib_logic
from lib_rebalance import get_lp_asset_qty_after_price_chg
import lib_rebalance as lib_reb


def get_portfolio_initial_value(starting_price, initial_qty_0_and_1 = None):
        if initial_qty_0_and_1 is None:
            initial_qty1 = 1 /2 
            initial_qty0 = initial_qty1 / starting_price
            
        else:
            if(len(initial_qty_0_and_1)!=2):
                raise ValueError("Input error: initial_qty_0_and_1 must have 2 elements!")
            initial_qty0 = initial_qty_0_and_1[0]
            initial_qty1 = initial_qty_0_and_1[1]
            
        return initial_qty0, initial_qty1

def portfolio_noLP_justhold(df, initial_qty_0_and_1 = None ):
    
    df.sort_index(ascending=True)
    p0 = df['price'].iat[0]
    pn = df['price'].iat[-1]
    
    qty0, qty1 = get_portfolio_initial_value(p0, initial_qty_0_and_1)

    array_col_names = [ 'end_p', 'end_qty0', 'end_qty1', 'end_value_base', 'end_value_quote', 'total_fee_yield',
        'begin_p', 'begin_qty0', 'begin_qty1', 'begin_value_base', 'begin_value_quote'
                 ]

    begin_value_base, begin_value_quote  = value_base_and_quote(qty0, qty1,p0)
    end_value_base, end_value_quote  = value_base_and_quote(qty0, qty1,pn)

    rst = np.array([ pn, qty0, qty1, end_value_base, end_value_quote,  0,
                    p0, qty0, qty1, begin_value_base, begin_value_quote ])
            

    df_rst = pd.DataFrame(data=[rst], columns=array_col_names)

    return df_rst 


def portfolio_norebalance_fixed_range(df, range_down, range_up, initial_qty_0_and_1 = None, benchmark_avg_yld_range = -0.1 ):
    
    df.sort_index(ascending=True)
    starting_price = df['price'].iat[0]
    end_price = df['price'].iat[-1]
    
    initial_qty0, initial_qty1 = get_portfolio_initial_value(starting_price, initial_qty_0_and_1)

    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, benchmark_avg_yld_range )

    p0 = starting_price
    pn = end_price
    
    qty0 = initial_qty0
    qty1 = initial_qty1

    end_qty0, end_qty1 = lib_reb.get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_down, range_up, b_input_pct = True)
        


    range_price_up = p0 * (1+range_up)
    range_price_down = p0 * (1+range_down)
    ps_b_within_range = df['price'].apply(lambda x: int(x>=range_price_down and x<=range_price_up ))
        
    df[ 'price_range_up'] = range_price_up
    df[ 'price_range_down'] = range_price_down
    df[ 'b_within_range'] =  ps_b_within_range


    fee_yield =boost_factor* (df['daily_fee_rate']* ps_b_within_range ).sum()
    end_qty0, end_qty1 = end_qty0 * (1+fee_yield), end_qty1*(1+fee_yield)

    array_col_names = [ 'end_p', 'end_qty0', 'end_qty1', 'end_value_base', 'end_value_quote', 'total_fee_yield',
        'begin_p', 'begin_qty0', 'begin_qty1', 'begin_value_base', 'begin_value_quote'
                 ]

    begin_value_base, begin_value_quote  = value_base_and_quote(qty0, qty1,p0)
    end_value_base, end_value_quote  = value_base_and_quote(end_qty0, end_qty1,pn)

    rst = np.array([ pn, end_qty0, end_qty1, end_value_base, end_value_quote,  fee_yield,
                    p0, qty0, qty1, begin_value_base, begin_value_quote ])
            

    df_rst = pd.DataFrame(data=[rst], columns=array_col_names)

    return df_rst 


def rebal_get_price_predict(df):
    df_p = df[['price']].copy()
    df_p.sort_index(ascending=True, inplace=True)

    df_p['1d_chg'] = np.log(df_p['price'] / df_p['price'].shift(1))
    df_p['1dchg_MA7'] = df_p['1d_chg'].rolling(window=7).mean()
    df_p['price_MA7'] = df_p['price'].rolling(window=14).mean()
    df_p['p_predict'] = df_p['price_MA7'].shift(1) * np.exp( df_p['1dchg_MA7'].shift(1) ) 
    return df_p['p_predict']

def rebal_between(p, up, down):
    if up < down:
        up, down = down, up
    return int(p<=up and p>=down)

def value_base_and_quote(qty0, qty1, p_token0):
    value_quote = qty0*p_token0 + qty1
    value_base = qty0 + qty1/p_token0
    return value_base, value_quote


def mav_pool_get_bin_pos(p_cur, price_bin_left_0, price_bin_right_0 ):
    mode = "middle"
    if p_cur < price_bin_left_0  : # describe where the bin sit vs price_current
        mode = "right"
    elif p_cur > price_bin_right_0 :
        mode = "left"
    return mode

def get_new_range_given_range_pos(p0, gap = None, width = None, range_pos = "middle"):
    
    if(gap is None):
        gap = p0*0.01
    if(width is None):
        width = p0*0.05
    
    if(range_pos == "both" ):
        range_left = p0 + width/2
    elif(range_pos == "left"):
        range_left = p0 -  gap - width
    elif(range_pos == "right"):
        range_left = p0 + gap 

    return range_left, range_left + width


def get_bin_price_limit_given_cur_p_and_mode(p_new, price_bin_left_0, price_bin_right_0, gap, width, rebalance_mode = "both"):
    cur_pos = mav_pool_get_bin_pos(p_new, price_bin_left_0, price_bin_right_0 )
    
    if cur_pos == "middle": # if cur price pos is middle, then no need change at all
        return price_bin_left_0, price_bin_right_0
    
    if rebalance_mode == "both":
        return get_new_range_given_range_pos(p_new,  gap, width, range_pos = cur_pos)    
    elif rebalance_mode == "left":
        if cur_pos == "right":
            return get_new_range_given_range_pos(p_new, range_pos =cur_pos) # this will move bin left to closet bin near  cur price
        else:
            return  price_bin_left_0, price_bin_right_0
    else: # rebalance_mode ==  "right"
        if cur_pos == "left":
            return get_new_range_given_range_pos(p_new, range_pos =cur_pos) # this will move bin left to closet bin near  cur price
        else:
            return  price_bin_left_0, price_bin_right_0



def mav_pool_get_earn_pct(string_bin_pos_0, string_bin_pos_new, p0, p_new, price_bin_left_0, price_bin_right_0):
    pct = 0
    case = string_bin_pos_0 + "-" +string_bin_pos_new
    bin_width = price_bin_right_0 - price_bin_left_0
    
    if case == 'middle-middle':
        pct = 1
    elif case == 'middle-left':
        pct = (price_bin_right_0 - p0) / (p_new - p0)
    elif case == 'middle-right':
        pct = (p0 - price_bin_left_0) / (p0-p_new)
    elif case == 'left-middle':
        pct = (price_bin_right_0 - p_new) / (p0-p_new)
    elif case == 'left-left':
        pct = 0
    elif case == 'left-right':
        pct = bin_width / (p0-p_new)
    elif case == 'right-middle':
        pct = (p_new - price_bin_left_0) / (p_new - p0)
    elif case == 'right-left':
        pct = bin_width / (p_new - p0)
    elif case == 'right-right':
        pct = 0
    else:
        print ("error in mav pool get_earn_pct")
        
    return pct


def portfolio_rebal_follow_p(df, range_down, range_up, initial_qty_0_and_1 = None, benchmark_avg_yld_range = -0.1 , follow_mode = 'both'  ):
    
    df.sort_index(ascending=True)

    starting_price = df['price'].iat[0]
    initial_qty0, initial_qty1 = get_portfolio_initial_value(starting_price, initial_qty_0_and_1)

    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, benchmark_avg_yld_range )

    array_col_names = ['date', 'qty0', 'qty1', 'price', 'value_base', 'value_quote', 'period_fee_yield',
                       'old_range_down', 'old_range_up', 'new_range_down', 'new_range_up']

    # Create an empty DataFrame with specified column names
    df_rst = pd.DataFrame(columns=array_col_names)


    p0 = starting_price
    qty0 = initial_qty0
    qty1 = initial_qty1
    
    p0_index = df.index[0]
    range_price_up = p0 * (1+range_up)
    range_price_down = p0 * (1+range_down)
    gap = 0.01*p0 #  
    width = range_price_up - range_price_down
    last_index = df.index[-1]
    
    
    value_base, value_quote =  value_base_and_quote(qty0, qty1, p0)
        
    df_rst.loc[len(df_rst)] = np.array([p0_index, qty0, qty1, p0, value_base, value_quote, 0,
                                    range_price_down, range_price_up, range_price_down, range_price_up ])
    
    
    range_pos_p0 = mav_pool_get_bin_pos(p0, range_price_down, range_price_up ) # start from middle
    
    for index, row in df.iloc[1:].iterrows():
        pn = row['price']
        range_pos_pn = mav_pool_get_bin_pos(pn, range_price_down, range_price_up ) 
    

        if(index != last_index and range_pos_p0 == range_pos_pn and range_pos_pn != "middle") :
            # price stays outside and didn't move in. we just need to update range limit
            range_price_down, range_price_up = get_new_range_given_range_pos(pn, gap = gap, width = width, range_pos = range_pos_p0)
            p0, p0_index =pn, index
            continue


        fee_rate_pct  = mav_pool_get_earn_pct(range_pos_p0, range_pos_pn, p0, pn, range_price_down, range_price_up)
        period_fee_yield =boost_factor* df['daily_fee_rate'].at[index] * fee_rate_pct
        
        end_qty0, end_qty1 = get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_price_down, range_price_up, b_input_pct = False)
        end_qty0, end_qty1 = end_qty0*(1+period_fee_yield), end_qty1*(1+period_fee_yield)
        
        
        range_price_down_n, range_price_up_n = get_bin_price_limit_given_cur_p_and_mode(pn, range_price_down, range_price_up, gap, width, rebalance_mode = follow_mode)
        p0_new = pn
    
        value_base, value_quote  =  value_base_and_quote(end_qty0, end_qty1, pn)
        
        df_rst.loc[len(df_rst)] = np.array([index, end_qty0, end_qty1, pn, value_base, value_quote, period_fee_yield,
                                    range_price_down, range_price_up, range_price_down_n, range_price_up_n ])


        p0, p0_index =p0_new, index
        qty0, qty1 = end_qty0, end_qty1 
        range_price_up, range_price_down = range_price_up_n, range_price_down_n 
        range_pos_p0 =  range_pos_pn


    rst_names = [ 'end_p', 'end_qty0', 'end_qty1', 'end_value_base', 'end_value_quote', 'total_fee_yield',
        'begin_p', 'begin_qty0', 'begin_qty1', 'begin_value_base', 'begin_value_quote'
                 ]
    begin_value_base, begin_value_quote   =value_base_and_quote(initial_qty0, initial_qty1, starting_price)
    total_fee_yield = df_rst['period_fee_yield'].sum()
    last_row = df_rst.iloc[-1]
    end_p, end_qty0, end_qty1, value_base, value_quote = last_row['price'], last_row['qty0'], last_row['qty1'], last_row['value_base'], last_row['value_quote'] 
    rst_values = [end_p, end_qty0, end_qty1, value_base, value_quote, total_fee_yield,
                    starting_price,initial_qty0,initial_qty1,  begin_value_base, begin_value_quote
                ]


    rst_summary = pd.DataFrame(data=[rst_values], columns=rst_names)
    return rst_summary, df_rst    



def portfolio_rebal_buylowsellhigh_predict(df, range_down, range_up, initial_qty_0_and_1 = None, benchmark_avg_yld_range = -0.1  ):
    
    df.sort_index(ascending=True)
    df['p_predict'] = rebal_get_price_predict(df)


    starting_price = df['price'].iat[0]
    initial_qty0, initial_qty1 = get_portfolio_initial_value(starting_price, initial_qty_0_and_1)


    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, benchmark_avg_yld_range )

    array_col_names = ['date', 'qty0', 'qty1', 'price', 'value_base', 'value_quote', 'period_fee_yield',
                       'old_range_down', 'old_range_up', 'new_range_down', 'new_range_up']

    # Create an empty DataFrame with specified column names
    df_rst = pd.DataFrame(columns=array_col_names)


    p0 = starting_price
    qty0 = initial_qty0
    qty1 = initial_qty1
    
    p0_index = df.index[0]
    range_price_up = p0 * (1+range_up)
    range_price_down = p0 * (1+range_down)
    shift_step = p0*0.02 #  
    period_days = 0
    last_index = df.index[-1]
    
    
    value_base, value_quote =  value_base_and_quote(qty0, qty1, p0)
        
    df_rst.loc[len(df_rst)] = np.array([p0_index, qty0, qty1, p0, value_base, value_quote, 0,
                                    range_price_down, range_price_up, range_price_down, range_price_up ])
    
    for index, row in df.iloc[1:].iterrows():
        pn = row['price']
        p_pred = row['p_predict']

        
        if(rebal_between(pn, range_price_up, range_price_down)):
            period_days=period_days+1
            continue
        if(period_days  == 0 and index != last_index) :
            continue 
        
        
        end_qty0, end_qty1 = get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_price_down, range_price_up, b_input_pct = False)
        

        # daily_yield * Lx/Lx+Ly * 2 * boost * (1, 0)
        prev_period_index = (df.index > p0_index) & (df.index <= index)
        ps_b_within_range = df.loc[prev_period_index, 'price'].apply(lambda x: rebal_between(x,range_price_up, range_price_down) )
        
        df.loc[prev_period_index, 'price_range_up'] = range_price_up
        df.loc[prev_period_index, 'price_range_down'] = range_price_down
        df.loc[prev_period_index, 'b_within_range'] =  ps_b_within_range

        period_fee_yield =boost_factor* ( df.loc[prev_period_index, 'daily_fee_rate'] * ps_b_within_range ).sum()
        
        end_qty0, end_qty1 = end_qty0*(1+period_fee_yield), end_qty1*(1+period_fee_yield)
        
        # the best case, range (down, up) is between current price and prediction price. 
        range_down_predict, range_up_predict = p_pred * (1+range_down), p_pred * (1+range_up)
        
        
        if (pn < range_price_down) : 
            range_price_down_n = max(range_price_down + shift_step, range_down_predict)
            range_price_up_n =  max(range_price_up + shift_step, range_up_predict)
            p0_new = range_price_down_n
            
        elif pn > range_price_up: # # pos:    range_down, range up, pn
            range_price_down_n = min(range_price_down - shift_step,    range_down_predict)  
            range_price_up_n = min(range_price_up - shift_step, range_up_predict)
            p0_new = range_price_up_n
            
        else:
            # ??? will this happen? or no? when last row, where price not moving outside
            range_price_up_n, range_price_down_n = range_price_up, range_price_down
            p0_new = pn # it is due to  the qty is reproduced by pn
    
        value_base, value_quote  =  value_base_and_quote(end_qty0, end_qty1, pn)
        
        df_rst.loc[len(df_rst)] = np.array([index, end_qty0, end_qty1, pn, value_base, value_quote, period_fee_yield,
                                    range_price_down, range_price_up, range_price_down_n, range_price_up_n ])


        p0, p0_index, period_days =p0_new, index, 0
        qty0, qty1 = end_qty0, end_qty1 
        range_price_up, range_price_down = range_price_up_n, range_price_down_n          


    rst_names = [ 'end_p', 'end_qty0', 'end_qty1', 'end_value_base', 'end_value_quote', 'total_fee_yield',
        'begin_p', 'begin_qty0', 'begin_qty1', 'begin_value_base', 'begin_value_quote'
                 ]
    begin_value_base, begin_value_quote   =value_base_and_quote(initial_qty0, initial_qty1, starting_price)
    total_fee_yield = df_rst['period_fee_yield'].sum()
    last_row = df_rst.iloc[-1]
    end_p, end_qty0, end_qty1, value_base, value_quote = last_row['price'], last_row['qty0'], last_row['qty1'], last_row['value_base'], last_row['value_quote'] 
    rst_values = [end_p, end_qty0, end_qty1, value_base, value_quote, total_fee_yield,
                    starting_price,initial_qty0,initial_qty1,  begin_value_base, begin_value_quote
                ]


    rst_summary = pd.DataFrame(data=[rst_values], columns=rst_names)
    return rst_summary, df_rst    

    # Create a dictionary using zip and dictionary comprehension
    # rst_dict = {name: value for name, value in zip(rst_names, rst_values)}

def portfolio_rebal_recentre(df, range_down, range_up, initial_qty_0_and_1 = None, benchmark_avg_yld_range = -0.1  ):
    
    df.sort_index(ascending=True)
    df['p_predict'] = rebal_get_price_predict(df)


    starting_price = df['price'].iat[0]
    initial_qty0, initial_qty1 = get_portfolio_initial_value(starting_price, initial_qty_0_and_1)


    boost_factor = lib_logic.get_liquidity_boost_given_range(range_down, benchmark_avg_yld_range )

    array_col_names = ['date', 'qty0', 'qty1', 'price', 'value_base', 'value_quote', 'period_fee_yield',
                       'old_range_down', 'old_range_up', 'new_range_down', 'new_range_up']

    # Create an empty DataFrame with specified column names
    df_rst = pd.DataFrame(columns=array_col_names)


    p0 = starting_price
    qty0 = initial_qty0
    qty1 = initial_qty1
    
    p0_index = df.index[0]
    range_price_up = p0 * (1+range_up)
    range_price_down = p0 * (1+range_down)
    shift_step = 0.0005 #  
    period_days = 0
    last_index = df.index[-1]
    
    
    value_base, value_quote =  value_base_and_quote(qty0, qty1, p0)
        
    df_rst.loc[len(df_rst)] = np.array([p0_index, qty0, qty1, p0, value_base, value_quote, 0,
                                    range_price_down, range_price_up, range_price_down, range_price_up ])
    
    for index, row in df.iloc[1:].iterrows():
        pn = row['price']


        
        if(rebal_between(pn, range_price_up, range_price_down) and index != last_index):
            period_days=period_days+1
            continue
        
        
        end_qty0, end_qty1 = get_lp_asset_qty_after_price_chg(p0, pn, qty0, qty1, range_price_down, range_price_up, b_input_pct = False)
        

        # daily_yield * Lx/Lx+Ly * 2 * boost * (1, 0)
        prev_period_index = (df.index > p0_index) & (df.index <= index)
        ps_b_within_range = df.loc[prev_period_index, 'price'].apply(lambda x: rebal_between(x,range_price_up, range_price_down) )
        
        df.loc[prev_period_index, 'price_range_up'] = range_price_up
        df.loc[prev_period_index, 'price_range_down'] = range_price_down
        df.loc[prev_period_index, 'b_within_range'] =  ps_b_within_range

        period_fee_yield =boost_factor* ( df.loc[prev_period_index, 'daily_fee_rate'] * ps_b_within_range ).sum()
        
        end_qty0, end_qty1 = end_qty0*(1+period_fee_yield), end_qty1*(1+period_fee_yield)
        
        
        # swap and reblance to centre. 
        if( rebal_between(pn, range_price_up, range_price_down) ):
            range_price_up_n = range_price_up
            range_price_down_n = range_price_down
        else:
            range_price_up_n = pn * (1+range_up)
            range_price_down_n = pn * (1+range_down)
            
            total_value_quote = end_qty0*pn + end_qty1
            end_qty0 = total_value_quote/2 / pn
            end_qty1 = total_value_quote/2
            

        # note: place this function after rebalance
        value_base, value_quote  =  value_base_and_quote(end_qty0, end_qty1, pn)
        
        df_rst.loc[len(df_rst)] = np.array([index, end_qty0, end_qty1, pn, value_base, value_quote, period_fee_yield,
                                    range_price_down, range_price_up, range_price_down_n, range_price_up_n ])


        p0, p0_index, period_days =pn, index, 0
        qty0, qty1 = end_qty0, end_qty1 
        range_price_up, range_price_down = range_price_up_n, range_price_down_n          


    rst_names = [ 'end_p', 'end_qty0', 'end_qty1', 'end_value_base', 'end_value_quote', 'total_fee_yield',
        'begin_p', 'begin_qty0', 'begin_qty1', 'begin_value_base', 'begin_value_quote'
                 ]
    begin_value_base, begin_value_quote   =value_base_and_quote(initial_qty0, initial_qty1, starting_price)
    total_fee_yield = df_rst['period_fee_yield'].sum()
    last_row = df_rst.iloc[-1]
    end_p, end_qty0, end_qty1, value_base, value_quote = last_row['price'], last_row['qty0'], last_row['qty1'], last_row['value_base'], last_row['value_quote'] 
    rst_values = [end_p, end_qty0, end_qty1, value_base, value_quote, total_fee_yield,
                    starting_price,initial_qty0,initial_qty1,  begin_value_base, begin_value_quote
                ]


    rst_summary = pd.DataFrame(data=[rst_values], columns=rst_names)
    return rst_summary, df_rst    

    # Create a dictionary using zip and dictionary comprehension
    # rst_dict = {name: value for name, value in zip(rst_names, rst_values)}



def get_performance_given_scenario(date_begin= '2021-12-09', date_end= '2022-09-09'):

    df_price = m.get_df_daily_price(date_begin,date_end)
    df_fee = m.get_df_daily_fees(date_begin, date_end)
    df = m.get_df_comb_price_fee(df_price, df_fee)
    print("\ndf date", max(df['date']), min(df['date']))

    df_noLP = portfolio_noLP_justhold(df)

    range_down = -0.15 # yearly no rebalance
    range_up = -1*range_down/(1+range_down) 
    df_noreb_fixed_range = portfolio_norebalance_fixed_range(df.copy(), range_down, range_up )

    range_down = -0.1 # recenter-methodology
    range_up = -1*range_down/(1+range_down) 
    df_rebal_recentre, df_rst = portfolio_rebal_recentre(df.copy(), range_down, range_up )

    range_down = -0.05
    range_up = -1*range_down/(1+range_down) 
    df_rebal_blsh, df_rst = portfolio_rebal_buylowsellhigh_predict(df.copy(), range_down, range_up )

    range_down = -0.01
    range_up = -1*range_down/(1+range_down)
    df_rebal_md_both, df_rst = portfolio_rebal_follow_p (df.copy(), range_down, range_up, follow_mode = 'both')
    df_rebal_md_right, df_rst = portfolio_rebal_follow_p (df.copy(), range_down, range_up, follow_mode = 'right')
    df_rebal_md_left, df_rst = portfolio_rebal_follow_p (df.copy(), range_down, range_up, follow_mode = 'left')


    df_1scen = pd.concat((df_noLP, df_noreb_fixed_range, df_rebal_recentre,  df_rebal_blsh, df_rebal_md_both, df_rebal_md_right, df_rebal_md_left))
    df_1scen.index = ['noLP', "fixed", "swap_recntr", "rg_blsh", 'md-both', 'md-right', 'md-left']

    return df_1scen[[ 'begin_value_quote','end_value_quote', 'total_fee_yield' ]]
    


df_scenarios = lib_reb.get_lp_evaluation_scenarios()
print(df_scenarios)

method_names = ['noLP', "fixed", "swap_recntr", "rg_blsh", 'md-both', 'md-right', 'md-left']
df_all_acen = pd.DataFrame(columns=method_names)
df_all_start = df_all_acen.copy()


for index, row in df_scenarios.iterrows():
    date_begin = row["date_begin"]
    date_end = row["date_end"]
    df_scen = get_performance_given_scenario(date_begin, date_end)
    df_all_acen.loc[len(df_all_acen)] = np.array(df_scen["end_value_quote"])
    df_all_start.loc[len(df_all_acen)] = np.array(df_scen["begin_value_quote"])

df_all_acen.index = df_scenarios['scenario_name']

df_divided = df_all_acen.apply(lambda row: row / row.iloc[0], axis=1)

print(df_all_start)
print(df_all_acen)
print(df_divided)

#df_all_acen.to_clipboard()