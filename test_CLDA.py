# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:57:00 2018

@author: slab
"""

import numpy as np
from matplotlib import pyplot as plt
import C_LDA
import tradition_LDA as t_LDA

start = 10
topics = start 
end = 1
iteration_num = 30
clip = 50
palpha = 0.05
pgamma = 0.05
pbeta = 0.05
c_len = 10

# TDT2 Dataset
tdt2_data = np.load("tdt2_em_v4_0_100.npy")
tdt2_data = tdt2_data[:clip]
stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')
texts = [[word for word in line.lower().split() if word not in stop_list] for line in tdt2_data] 
t_data = texts[:clip]

save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
# RUN C-LDA
C_LDA.run(t_data, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)

dataset = save_p
y1 = np.load(str(dataset) +"\\C-LDAper_list"+ str(topics) +".npy")
x = np.linspace(0, iteration_num, iteration_num)
plt.plot(x[::1], y1[:], "r*-", label='C-LDA', linewidth=1)
plt.title("Convergence Test By Perplexities")
plt.ylabel(u"Perplexities")
plt.xlabel(u"Iterations") 
plt.legend(loc="upper right")
plt.show()

save_p2 = "tradition_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
t_LDA.run(t_data, start, end, iteration_num, save_p2, clip, c_len, palpha, pbeta, pgamma)

dataset = save_p2
y2 = np.load(str(dataset) +"\\LDAper_list"+ str(topics) +".npy")
x = np.linspace(0, iteration_num, iteration_num)
plt.plot(x[::1], y1[:], "r*-", label='C-LDA', linewidth=1)
plt.plot(x[::1], y2[:], "b+-", label='LDA', linewidth=1)
plt.title("Convergence Test By Perplexities")
plt.ylabel(u"Perplexities")
plt.xlabel(u"Iterations") 
plt.legend(loc="upper right")
plt.show()
