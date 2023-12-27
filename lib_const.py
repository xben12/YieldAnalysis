

pool_info_list = [
    ['0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', 'WBTC',  'WETH',  0.003],
    ['0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' , 'USDC', 'WETH', ],
    ['0x5777d92f208679db4b9778590fa3cab3ac9e2168' , 'DAI', 'USDC', ], 
    ['0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa', 'WSTETH', 'WETH',],
    ['0x4585fe77225b41b697c938b018e2ac67ac5a20c0' , 'WBTC' ,'WETH', 0.0005]
]

# data needed by coingecko API
price_token_list =[
    ['bitcoin', 'BTC' ],
    ['ethereum',  'ETH']
]

def get_pool_filename(pool_address, token0=None, token1=None):    
    if token0 is None or token1 is None:
        for pool_info in pool_info_list:
            if pool_info[0] == pool_address:
                token0 = pool_info[1]
                token1 = pool_info[2]
                break
        
    file_name_addon = "_".join([pool_address, token0, token1])
    file_name = 'output/pool_data_' + file_name_addon + '.csv'
    return file_name

def get_crypto_price_filename(token = None):
    return 'output/price_data_all_token.csv'

#    date_begin = datetime.strptime(date_begin_yyyymmdd, '%Y%m%d')
#    date_end = datetime.strptime(date_end_yyyymmdd, '%Y%m%d')