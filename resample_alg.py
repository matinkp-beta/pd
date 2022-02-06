import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import os
import platform


from scipy.io import wavfile 

if platform.system() == "Linux":
    clear = "clear"
else: clear = "cls"


def play_notification():
    fs, audio = wavfile.read('../alarm.wav')
    sd.play(audio, fs)

def update_progress(progress):
    bar_length = 20
    block = int(round(bar_length * progress))

    os.system(clear)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def get_min_length(address: str ,res_num: int ):
    os.system(clear)
    print("loading...")
    url_list= [f"./{address}/res_{i}.txt" for i in range(1,res_num+1)]
    dataframe =[]
    j=0
    tempForSize =[]
    time = []
    for url in url_list:
        counter = 0
        data = []
        df = pd.read_csv(url, sep=r'\s{2,}', engine='python', header=None, names=['time','port'])
        df1=df.drop([0,1])
        df1.reset_index(drop=True,inplace=True)
    
        #normalize data 
        df1["port"]=np.float32(df1["port"])/abs((np.float32(df1["port"]))).max()

        #cut start data
        for x in np.float32(df1["port"]):
            counter+=1
            if abs(x)>= 0.001:
                df2=df1.drop(range(counter))
                df2["time"] = np.float32(df2["time"]) - np.float32(df1.loc[counter]["time"])
                time.append(float(df1.loc[counter-1]["time"]))
                break
        df2.reset_index(drop=True,inplace=True)
        dataframe.append(df2.squeeze())
        j+=1
        update_progress(j/res_num)
    max_time = 50.0 - max(time)
    return max_time,dataframe


def cut_time_and_min_sample(max_time:float,data,res_num: int):
    os.system(clear)
    tempForSize=[]
    dataframe = []
    j=0
    for datum in data:
        df2=datum.drop( datum.loc[datum["time"] >= max_time].index )

        l,t=df2.shape
        dataframe.append(df2)
        tempForSize.append(l)
        j+=1
        update_progress(j/res_num)
    return min(tempForSize),dataframe


def create_dataframe(sample_rate:int,data,res_num: int):
    dataframe = pd.DataFrame(columns=range(sample_rate))
    j=0
    for datum in data:
        temp=[]
        l,t=datum.shape
        samples = np.linspace(0, l-1, num=sample_rate,dtype = int)
        for i in samples:
            normData= np.float32(datum.loc[i]["port"])
            temp.append(normData)
        dataframe.loc[j]=np.array(temp)
        j+=1
        update_progress(j/res_num)
    print(dataframe)
    play_notification()
    return dataframe

address = "data"
num_of_res = 500
max_time,data=get_min_length(address,num_of_res)
sample_rate,data=cut_time_and_min_sample(max_time,data,num_of_res)
dataframe = create_dataframe(sample_rate,data,num_of_res)
os.system(clear)
dataframe.head()