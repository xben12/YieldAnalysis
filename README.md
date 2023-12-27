# how to run it
* poetry install
* poetry run python main.py (get monthly optimal yield with trading off between capital efficiency and imp loss)
* poetry run python main_avg_yld.py (range to get average yeild)

# detail
methodology explanation can be found in medium article: 
* https://medium.com/@xben12/defi-decode-liquidity-mining-yield-impermanent-loss-and-set-optimal-range-e20c3472d2bb
* 

Code: 
* main: main.py, main_avg_yld.py
* data: use library_data.py. This file will pre save data into csv in the output folder.
* logic (imp loss, range, coverage %): library_logic.py