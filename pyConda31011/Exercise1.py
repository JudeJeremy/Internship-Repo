"""Exercise 1: 1.Write a Pandas program to create a 
dataframe from a dictionary and display it.
Sample data: {'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]}
"""
import pandas as pd


sampledata = {
    'X':[78,85,96,80,86], 
    'Y':[84,94,89,83,86],
    'Z':[86,97,96,72,83],
}


dataframe1 = pd.DataFrame(sampledata)
dataframe1

"""Exercise Complete"""