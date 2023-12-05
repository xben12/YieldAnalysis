# function for full range LP
import numpy as np
import pandas as pd

# Get impermanent loss, qty_change_token0, and token1. 
# code also works if input is array
def get_impermanent_loss(price_change, is_change_token0=True, b_return_df = False):
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


# Check if the script is being run as the main program
if __name__ == "__main__":
    result_matrix = get_impermanent_loss(0.1)
    print(result_matrix)
    result_matrix[1]
    result_matrix = get_impermanent_loss(0.1,  b_return_df=True)
    print(result_matrix)
    result_matrix = get_impermanent_loss([0.1, 0.5, 1])
    print(result_matrix)
    result_matrix = get_impermanent_loss([0.1, 0.5, 1], b_return_df=True)
    print(result_matrix)