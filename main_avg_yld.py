import numpy as np
import lib_logic as lib_lgc
import main as m

def create_lp_by_pos_by_lin_interp(range_pos_price_down, range_pos_lp_qty,  step = -0.01):

    start = max(range_pos_price_down)
    stop = min(range_pos_price_down)


    range_down = np.arange(start, stop + step, step)
    range_down_lp_qty = np.empty(len(range_down))

    for i in range(1,len(range_pos_price_down)):
        pos_begin = int(range_pos_price_down[i-1] / step) -1
        pos_end = int(range_pos_price_down[i] / step) -1
        
        # value 
        value_start = range_pos_lp_qty[i-1]
        value_end = range_pos_lp_qty[i]
        value_step = (value_end -value_start ) /(pos_end -pos_begin )
        rst =   np.arange(value_start, value_end+value_step/2, value_step)

        range_down_lp_qty[pos_begin:(pos_end+1)] =rst

    return range_down, range_down_lp_qty

def get_all_range_coverage_rate_monthly(range_down, df):

    df_price_chg = df[['Price_chg_vs_MM01' ]][df['month_begin_date'] != df['date']]
    total_sample_count = len(df_price_chg)
    rst = np.empty(len(range_down))
    for i in range(0, len(range_down)):
        range_d =  range_down[i]
        range_u = lib_lgc.get_opposite_bin_limit_with_same_liquidity(range_d)
        
        count_within_range =df_price_chg['Price_chg_vs_MM01'].between(range_d, range_u).sum()
        coverage_rate= count_within_range / total_sample_count
        rst[i] = coverage_rate

    return rst


# these are inflection points of liqudity, the value must be multiples of -0.01 (or -1%). 
range_pos_price_down = np.array([-0.01, -0.05, -0.2])
range_pos_lp_qty = np.array([2, 0.66, 0.29])

range_down, range_down_lp_qty = create_lp_by_pos_by_lin_interp(range_pos_price_down, range_pos_lp_qty)

df = m.get_df_daily_price(date_begin = '2022-12-01', date_end ="2023-11-30")
coverage_rate = get_all_range_coverage_rate_monthly(range_down, df)


w_tvl = range_down_lp_qty / range_down_lp_qty.sum()
w_tvl_cvg = w_tvl*coverage_rate 
w_tvl_cvg = w_tvl_cvg/ w_tvl_cvg.sum() # normalisation


avg_d_tvl_only  =  1/sum(w_tvl / range_down)
avg_d_tvl_coverage = 1/   sum(w_tvl_cvg / range_down)
print("Range down limit to get market average yield: ", avg_d_tvl_coverage)
print("Range down limit to get market average yield when ignore rang e cross effect (not accurate): ", avg_d_tvl_only)


import matplotlib.pyplot as plt

x = np.concatenate((np.flip(range_down), -1*range_down))
y = np.concatenate((np.flip(range_down_lp_qty), range_down_lp_qty))

print("show range asset qty. ")
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Range (bin) Asset amount vs range_down value')
plt.xlabel('range_down')
plt.ylabel('Range Asset amount')
plt.ylim(0, 7)
plt.grid(True)
plt.show()

print("show coverage rate. ")
plt.plot(range_down, coverage_rate, marker='o', linestyle='-')
plt.title('coverage_rate vs range_down value')
plt.xlabel('range_down')
plt.ylabel('coverage_rate')

plt.grid(True)
plt.show()

print("show range weight. ")
plt.plot(range_down, w_tvl_cvg, label='weight w coverage', marker='o', linestyle='-')
plt.plot(range_down, w_tvl, label='weight w/o coverage', marker='s', linestyle='--')

plt.title('Comparison of weight with/without coverage')
plt.xlabel('range down limit')
plt.ylabel('normalised weight')

plt.legend()
plt.grid(True)
plt.show()