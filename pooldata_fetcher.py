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

# Replace 'YOUR_POOL_ADDRESS', 'FROM_TIMESTAMP', and 'TO_TIMESTAMP' with your actual values
# pool_address = '0xcbcdf9626bc03e24f779434178a73a0b4bad62ed' #WBTC WETH 0.3%
# pool_address = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640' #USDC WETH
# pool_address = '0x5777d92f208679db4b9778590fa3cab3ac9e2168' #DAI USDC
# pool_address = '0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa' #WSTETH WETH
pool_address = '0x4585fe77225b41b697c938b018e2ac67ac5a20c0' #WBTC WETH, 0.05%
years = [2022, 2023]
# Display the DataFrame
result_df = get_uniswap_v3_data_year(pool_address, years)
result_df.to_csv('output_' + pool_address + '.csv', index=False)
