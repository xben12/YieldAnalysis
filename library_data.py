import requests
from datetime import datetime
import pandas as pd

# API each time can only get 100 recoreds, hence break down the retrieve into year-month
def get_uniswap_v3_data_limit100(pool_address, from_timestamp, to_timestamp):
    # Uniswap V3 Subgraph endpoint
    endpoint = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

    # GraphQL query to get historical data
    query = '''
    {
    poolDayDatas( orderBy: date, 
    where: { pool: "%s", date_gte: %d, date_lt: %d } ) {
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
    print(query)
    # Make the GraphQL request
    response = requests.post(endpoint, json={'query': query})
    data = response.json()
    #print(data)
    return data['data']['poolDayDatas']

def get_uniswap_v3_data_year(pool_address, years):
    pool_df=pd.DataFrame()

    # Get Uniswap V3 data
    for year in years:
        for month in range(1, 13):
            from_timestamp = int(datetime(year, month, 1).timestamp())  # Replace with your start date
            if(month<12):
                to_timestamp_exclude = int(datetime(year, month+1, 1).timestamp())    # Replace with your end date
            else:
                to_timestamp_exclude = int(datetime(year+1, 1, 1).timestamp())
            uniswap_v3_data_month = get_uniswap_v3_data_limit100(pool_address, from_timestamp, to_timestamp_exclude)

            df = pd.DataFrame(uniswap_v3_data_month)
            pool_df = pd.concat([pool_df, df], ignore_index=True)
    return pool_df


import requests
from datetime import datetime, timedelta
import pandas as pd

def get_crypto_price(symbol, start_date, end_date, vs_currency='usd'):
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
        return prices
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


# Function to add data to the DataFrame
def add_to_price_dataframe(df, data, token, vs_currency='usd'):
    for price in data:
        timestamp, value = price
        dt_object = datetime.utcfromtimestamp(timestamp / 1000)
        date_str = dt_object.strftime('%Y-%m-%d')
        df.loc[len(df)] = [date_str, token, vs_currency, value, value, value, value]





# Check if the script is being run as the main program
if __name__ == "__main__":
    load_pool_related_data = False;
    if load_pool_related_data: # getting pool fee/vol related data
        # Replace 'YOUR_POOL_ADDRESS', 'FROM_TIMESTAMP', and 'TO_TIMESTAMP' with your actual values
        # pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed' #WBTC WETH 0.3%
        # pool_address = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' #USDC WETH
        # pool_address = '0x5777d92f208679db4b9778590fa3cab3ac9e2168' #DAI USDC
        # pool_address = '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa' #WSTETH WETH
        pool_address = '0x4585fe77225b41b697c938b018e2ac67ac5a20c0' #WBTC WETH, 0.05%
        years = [2022, 2023]
        # Display the DataFrame
        result_df = get_uniswap_v3_data_year(pool_address, years)
        result_df.to_csv('output/output_' + pool_address + '.csv', index=False)


    load_price_related_data = True;
    if load_price_related_data: # getting pool fee/vol related data
        # Set the start and end date
        start_date = datetime(2020, 11, 1)
        end_date = start_date + timedelta(days=1)

        # Get daily prices for BTC (Bitcoin), ETH (Ethereum) in USD, and ETH in terms of BTC
        btc_prices = get_crypto_price('bitcoin', start_date, end_date)
        eth_prices_usd = get_crypto_price('ethereum', start_date, end_date)
        eth_prices_btc = get_crypto_price('ethereum', start_date, end_date, vs_currency='btc')

        # Create empty DataFrame
        df = pd.DataFrame(columns=['Date', 'Token', 'vs_currency', 'Open', 'Close', 'High', 'Low'])


        # Add BTC, ETH in USD, and ETH in terms of BTC data to the DataFrame
        if btc_prices:
            add_to_price_dataframe(df, btc_prices, 'BTC')
        if eth_prices_usd:
            add_to_price_dataframe(df, eth_prices_usd, 'ETH')
        if eth_prices_btc:
            add_to_price_dataframe(df, eth_prices_btc, 'ETH', vs_currency='btc')

        # Save the DataFrame to a CSV file
        df.to_csv('output/crypto_prices_with_currency.csv', index=False)

        print("DataFrame saved to 'output/crypto_prices_with_currency.csv'")
