import numpy as np 

prices=np.array([10.5, 23.99, 5.75, 99.95])
discount=10.0

final_prices=prices - (prices*discount / 100)
print("Original prices:", prices)
print("Final prices after", discount, "% discount:", final_prices)

