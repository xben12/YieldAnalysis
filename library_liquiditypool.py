# function for full range LP
import numpy as np
import pandas as pd

# Get impermanent loss, qty_change_token0, and token1. 
# code also works if input is array
def get_impermanent_loss(price_change, is_change_token0=True, b_return_df = False, ret_imp_loss_only = False ):
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


def get_price_range_same_liquidity(price_change):
    return -price_change/(1+price_change)

def get_liquidity_boost_by_range(prince_range, benchmark = -0.5):
    if(benchmark > 0):
        benchmark = get_price_range_same_liquidity(benchmark)

    boost  = (np.sqrt(1+benchmark) -1 )  / (np.sqrt(1+prince_range) -1 )
    return boost


def get_impermanent_loss_range_pos(price_change, price_range_down, is_change_token0=True, b_return_df = False):
    #input value check ignored here
    price_range_up = get_price_range_same_liquidity(price_range_down)

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
    loss = get_impermanent_loss_range_pos(price_change, price_range_down=-0.16)
    print("imp loss",price_change, loss)

    
    price_change = -0.08
    loss = get_impermanent_loss_range_pos(price_change, price_range_down=0.16)
    print("imp loss, all y gone",price_change, loss)

    your_range = -0.05
    benchmark = -0.15
    print('boost:',your_range,benchmark, "by", get_liquidity_boost_by_range (your_range,benchmark))

    your_range =np.array([-0.05, -0.1]) 
    benchmark = -0.15
    print('boost:',your_range,benchmark, "by", get_liquidity_boost_by_range (your_range,benchmark))