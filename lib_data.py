import requests
from datetime import datetime, timedelta
import pandas as pd


# API each time can only get 100 recoreds, hence break down the retrieve into year-month
def get_uniswap_v3_data_limit100(pool_address, from_timestamp, to_timestamp):
    # Uniswap V3 Subgraph endpoint
    endpoint = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

    # GraphQL query to get historical data
    query = '''
    {
    poolDayDatas( orderBy: date, 
    where: { pool: "%s", date_gte: %d, date_lte: %d } ) {
        date
        liquidity
        sqrtPrice
        token0Price
        token1Price
        volumeToken0
        volumeToken1
        feesUSD
      	volumeUSD
      	tvlUSD
    }
    }

    ''' % (pool_address, from_timestamp, to_timestamp)
    # print(query)
    # Make the GraphQL request
    response = requests.post(endpoint, json={'query': query})
    data = response.json()
    #print(data)
    return data['data']['poolDayDatas']

def last_day_of_month(year, month):
    # Calculate the first day of the next month
    first_day_of_next_month = datetime(year, month, 1) + timedelta(days=32)

    # Subtract one day to get the last day of the current month
    last_day_of_month = first_day_of_next_month.replace(day=1) - timedelta(days=1)

    return last_day_of_month


def get_uniswap_v3_data_year(pool_address, years):
    pool_df=pd.DataFrame()

    # Get Uniswap V3 data
    for year in years:
        for month in range(1, 13):
            
            from_timestamp = int(datetime(year, month, 1).timestamp())  # Replace with your start date
            next_mon_first_day = last_day_of_month(year, month)+ timedelta(days=1)
            to_timestamp = int(next_mon_first_day.timestamp())
            uniswap_v3_data_month = get_uniswap_v3_data_limit100(pool_address, from_timestamp, to_timestamp)
            df = pd.DataFrame(uniswap_v3_data_month)
            pool_df = pd.concat([pool_df, df], ignore_index=True)
            
    return pool_df

def get_crypto_price(symbol, token, start_date, end_date, vs_currency='usd'):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp()),
        'interval': 'daily',
        'days': 1200
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        prices = data.get('prices', [])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    columns = ['date', 'price']
    df = pd.DataFrame(prices, columns=columns) 
    df.drop(df.index[-1], inplace=True) # last date has 2 records
    
    df['date'] = df['date'].apply( lambda x: datetime.utcfromtimestamp(x / 1000).date()   )
    df['token'] = token
    df['vs_currency'] = vs_currency
    
    return df
    




# Check if the script is being run as the main program
if __name__ == "__main__":
    
    import lib_const
    
    load_all_pool_related_data = True
    if load_all_pool_related_data: # getting pool fee/vol related data
        
        years = [2022, 2023]
        
        result_df = pd.DataFrame()
        for pool_info in lib_const.pool_info_list:
            pool_address = pool_info[0]
            result_df = get_uniswap_v3_data_year(pool_address, years)
            file_name = lib_const.get_pool_filename(pool_address, token0=pool_info[1], token1=pool_info[2])
            print("save data:",file_name )
            result_df.to_csv(file_name, index=False)
            #print("will only run for the first pool during test. ")
            #break
    
    # if only wanna run individual pool    
    # pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed' #WBTC WETH 0.3%
    # pool_address = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' #USDC WETH
    # pool_address = '0x5777d92f208679db4b9778590fa3cab3ac9e2168' #DAI USDC
    # pool_address = '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa' #WSTETH WETH
    # pool_address = '0x4585fe77225b41b697c938b018e2ac67ac5a20c0' #WBTC WETH, 0.05%
    # result_df = get_uniswap_v3_data_year(pool_address, years)
    # file_name = lib_const.get_pool_filename(pool_address)
    # result_df.to_csv(file_name, index=False)
    # Display the DataFrame
    # result_df.head()


    load_price_related_data = False;
    if load_price_related_data: # getting pool fee/vol related data
        # Set the start and end date
        start_date = datetime(2020, 11, 1)
        end_date = start_date + timedelta(days=1)


        df = pd.DataFrame()
        for token in lib_const.price_token_list:
            token_name = token[0]
            token_ticker = token[1] 
            df_price = get_crypto_price(token_name, token_ticker , start_date, end_date)
            print(f'get token {token_ticker} price in usd' )
            df_price_btc = pd.DataFrame()
            if(token_ticker != 'BTC'):
                df_price_btc = get_crypto_price(token_name, token_ticker , start_date, end_date, vs_currency='btc')
                print(f'get token {token_ticker} price in btc' )
                
            df = pd.concat([df, df_price, df_price_btc], ignore_index=True)

        price_file_name = 'output/price_data_all_token.csv'
        df.to_csv(price_file_name, index=False)

        print(f"DataFrame saved to {price_file_name}")
