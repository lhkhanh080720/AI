from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Input Data
def Read_Data():
    df = pd.read_csv('D:\HCMUT\AI\HW\Data_KMeans.csv')
    #PLot
    # plt.scatter(df.Age, df['Income($)'], color='black')
    # plt.show()
    return df
    
#K-means++
def KMeans_plus2(df):
    centroids = []
    #Select 1 centroid randomly in data
    index = random.randint(0, 22)
    centroids.append([df.Age[index], df['Income($)'][index]])
    #Plot
    # plt.scatter(df.Age, df['Income($)'], s=40, color='black')
    # plt.scatter(df.Age[index], df['Income($)'][index], s=40, color='red')
    # plt.show()

#Main
if __name__ == "__main__":
    data = Read_Data()
    KMeans_plus2(data)