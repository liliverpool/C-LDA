# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:06:50 2018

@author: li wenbo
"""

import pymc3 as pm
import numpy as np
import math
from pprint import pprint
from matplotlib import pyplot as plt
import time
import os

def create_dictionary(data):
    global word_index, index_word
    for doc in data:
        for w in doc:
            if w not in word_index:
                word_index[w] = len(word_index)
    index_word = dict(zip(word_index.values(), word_index.keys()))



stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')
model_name = "C-LDA"
start = 9
end = 1
data_clip = 100
data = []
word_index = dict()
index_word = dict()
#create_dictionary(data) 
docs_num = 1
topic_num = 1
words_num = 1
alpha = 0.05
beta = 0.05
gamma = 0.05
context_len = 10
iteration_num = 30
topic_word = 0*np.ones([1, 1])
topic_word_list = 0*np.ones([1, 1, 1])
doc_topic = 0*np.ones([1, 1])
words_co_topic_list =  0*np.ones([1, 1, 1])
docs_list = []
doc_topic_distributions = []
topic_word_distributions = []
topic_word_distribution = []
perplexities = []
per_list = []
st= 0
ed= 0
total_time = 0

def compute_words_co_topic_list(c_len):
    global words_co_topic_list
    for d in docs_list:
        for i in range(0, len(d)):
            bottom = max(i - c_len, 0)
            upper = min(i + c_len + 1, len(d))
            for j in range(bottom, upper):
                if(i != j):
                    words_co_topic_list[d[j][1]][d[i][0]][d[j][0]] += 1

def get_a_topic(doc_topic_distribution):
    topics = np.random.multinomial(1, doc_topic_distribution)
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

def get_a_topic_old(doc_topic_distribution):
    z = pm.distributions.multivariate.Multinomial.dist(1, doc_topic_distribution)
    topics = z.random()
    topic = -1
    for i in range(0, len(topics)):
        if topics[i] > 0:
            topic = i
            break
    return topic

def initialize_distributions():
    global doc_topic_distributions, topic_word_distributions, topic_word_distribution
    doc_topic_distributions.clear()
    topic_word_distributions.clear()
    topic_word_distribution.clear()
    for i in range(0, docs_num):
        doc_topic_distributions.append(1./topic_num*np.ones([topic_num]))
        topics_pdf = [] 
        for j in range(0, topic_num):
            topics_pdf.append(1./words_num*np.ones([words_num]))
        topic_word_distributions.append(topics_pdf)
    for i in range(0, topic_num):
        topic_word_distribution.append(1./words_num*np.ones([words_num]))
    return

def initial_docs_list():
    global data, docs_list
    docs_list.clear()
    for doc in data:
         docs_list.append(np.ones([len(doc), 2], dtype = np.uint8))
    return

def initialize_values_docs_list():
    global docs_list
    for d in range(0, len(data)):
        for w in range(0, len(data[d])):
           docs_list[d][w] = [word_index[data[d][w]], get_a_topic(doc_topic_distributions[d])]
    return

def compute_doc_topic():
    global doc_topic
    doc_topic = np.array(doc_topic)
    doc_topic = 0*doc_topic
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            doc_topic[i][docs_list[i][j][1]] += 1

def compute_doc_topic_doc(d):
    global doc_topic
    doc_topic[d] = np.array(doc_topic[d])
    doc_topic[d] = 0*doc_topic[d]
    for j in range(0, len(docs_list[d])):
        doc_topic[d][docs_list[d][j][1]] += 1

def compute_topic_word():
    global topic_word
    topic_word = np.array(topic_word)
    topic_word = 0*topic_word
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word[docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

def compute_topic_word_list_doc(d):  
    global docs_list
    topic_word_list[d] = np.array(topic_word_list[d])
    topic_word_list[d] = 0*topic_word_list[d]
    for i in range(len(docs_list)):
        for j in range(0, len(docs_list[i])):
            topic_word_list[d][docs_list[i][j][1]][docs_list[i][j][0]] += 1
    return

def get_n_d_k(d, w, k):
    n_d_k = 0
    for i in range(0, len(docs_list[d])):
        if(i != w and docs_list[d][i][1]- k == 0):
            n_d_k += 1
    return n_d_k

#
def get_n_w_k(d, w, k):
    n_w_k = 0
    if(docs_list[d][w][1] - k == 0 and topic_word[k][docs_list[d][w][0]] > 0):
        n_w_k = topic_word[k][docs_list[d][w][0]] - 1
    else:
        n_w_k = topic_word[k][docs_list[d][w][0]]
    return n_w_k

# 
def get_total_n_k(d, w, k):
    total_n_k = np.sum(topic_word[k])
    if(docs_list[d][w][1] - k == 0):
        total_n_k = total_n_k - 1
    return total_n_k

#
def get_context_num_w2(text, w1, w2, k, c_len):
    indexes = [x for x,a in enumerate(text) if a[0] == w1]
    w2_list = []
    for i in indexes:
        bottom = max(i - c_len, 0)
        upper = min(i + c_len + 1, len(text))
        for j in range(bottom, upper):
            if(text[j][0] == w2 and text[j][1] == k and j!= i and j not in w2_list):
                w2_list.append(j)
    return len(w2_list)

def get_context_num_all(text, w1, k, c_len):
    indexes = [x for x,a in enumerate(text) if a[0] == w1]
    w_list = []
    for i in indexes:
        bottom = max(i - c_len, 0)
        upper = min(i + c_len + 1, len(text))
        for j in range(bottom, upper):
            if(text[j][1] == k and j!= i and j not in w_list):
                w_list.append(j)
    return len(w_list)
    
def get_context(d, w, c_len):
    bottom = max(w - c_len, 0)
    upper = min(w + c_len + 1, len(docs_list[d]))
    result = []
    for w in range(bottom, upper):
        if(docs_list[d][w][0] not in result):
            result.append(docs_list[d][w][0])
    return result
    

def compute_dominator(context_words, k, c_len):
    result = 0
    for doc in docs_list:
        for w1 in context_words:
            result += get_context_num_all(doc, w1, k, c_len)
    return result

def compute_numerator(context_words, w2, k, c_len):
    result = 0
    for doc in docs_list:
        for w1 in context_words:
            result += get_context_num_w2(doc, w1, w2, k, c_len)
    return result

def compute_dominator2(context_words, w2, k, c_len):
    res = np.zeros([topic_num])
    for w in context_words:
        res += words_co_topic_list[:,w,w2]
    return res.sum()

def compute_numerator2(context_words, w2, k, c_len):
    res = 0
    for w in context_words:
        res += words_co_topic_list[k][w][w2]
    return res

#
def recompute_w_topic_distribution(d, w):
    new_topic_distribution = np.ones([topic_num])
    for topic in range(0, topic_num):
        n_d_k = get_n_d_k(d, w, topic)
        n_w_k = get_n_w_k(d, w, topic)
        total_n_k = get_total_n_k(d, w, topic)
        context_words = get_context(d, w, context_len)
        numerator = compute_numerator2(context_words, docs_list[d][w][0], topic, context_len)
#        dominator = compute_dominator2(context_words, w, topic, context_len)
        p_d_w_k = ((n_d_k + alpha) +  (numerator + gamma))*(n_w_k + beta)/(total_n_k + words_num*beta) 
        new_topic_distribution[topic] = p_d_w_k
    new_topic_distribution = new_topic_distribution/new_topic_distribution.sum()   
    return new_topic_distribution
    
#
def gibbs_sampling():
    global doc_topic_distributions, eta_list, gamma_list, st, ed, total_time
    st = 0
    ed = 0
    total_time = 0
    for d in range(0, len(docs_list)):        
        st = time.time()
        for w in range(0, len(docs_list[d])):
            new_pdf = recompute_w_topic_distribution(d, w)
#            print(new_pdf)
            new_topic = get_a_topic(new_pdf)
            docs_list[d][w][1] = new_topic
        ed = time.time()
        total_time += ed - st

           
def recompute_distributions():
    compute_words_co_topic_list(context_len)
    for d in range(0, len(doc_topic)):
        doc_topic_distributions[d] = (doc_topic[d] + alpha) / (np.sum(doc_topic[d]) + len(doc_topic[d]) * alpha)
    for topic in range(0, len(topic_word)):
        topic_word_distribution[topic] = (topic_word[topic] + beta) / (np.sum(topic_word[topic]) + len(topic_word[topic]) * beta)


def compute_perplexities():
    global doc_topic_distributions, docs_list, topic_word_distribution, words_co_topic_list

    total = 0
    total_num = 0
    for d in range(0, len(docs_list)):
        for v in range(0, len(docs_list[d])):
            total_t = 0
            for k in range(0, len(topic_word_distribution)):
                w = docs_list[d][v][0]
                context_words = get_context(d, v, context_len)
                numerator = compute_numerator2(context_words, docs_list[d][v][0], k, context_len)
                dominator = compute_dominator2(context_words, docs_list[d][v][0], k, context_len)                   
#                p_d_w_k = (topic_word_distribution[k]+ (numerator + gamma)/(dominator + words_num*gamma)) /2
                p_d_w_k = topic_word_distribution[k][w]
                theta_d_k = (doc_topic_distributions[d][k]+ (numerator + gamma)/(dominator + words_num*gamma)) /2
#                theta_d_k = doc_topic_distributions[d][k]
                total_t += theta_d_k*p_d_w_k
            total_num += 1.0
            total += (-1)*math.log(total_t)
    
    return math.exp(total / total_num) 
        
        
def parameter_estimation():
    per_list.clear()
    print(model_name)
    for i in range(0, iteration_num):    
        gibbs_sampling()
        print(model_name + "_Iteration" , i, " time:  ", total_time)
        recompute_distributions()
        compute_doc_topic()
        compute_topic_word()     
        per_list.append(compute_perplexities())
    return
        
def save_result(path):
    if not os.path.exists(path):
        os.makedirs(path)
    LDA_docs_list = np.array(docs_list) 
    LDA_doc_topic_distributions = np.array(doc_topic_distributions)
    LDA_topic_word_distribution = np.array(topic_word_distribution)
    np.save(path + str(model_name)+"docs_list"+str(topic_num)+".npy", LDA_docs_list)
    np.save(path + str(model_name)+"doc_topic_distributions_"+str(topic_num)+".npy", LDA_doc_topic_distributions)
    np.save(path + str(model_name)+"topic_word_distribution_"+str(topic_num)+".npy", LDA_topic_word_distribution)
    np.save(path + str(model_name)+"words_co_topic_list_"+str(topic_num)+".npy", words_co_topic_list)
    LDA_per_list = np.array(per_list)
    np.save(path + str(model_name)+"per_list"+str(topic_num)+".npy", LDA_per_list)

def initialize():
    global topic_word, doc_topic, topic_word_list, words_co_topic_list
    print("initializing...")
    topic_word = 0*np.ones([topic_num, words_num])
    doc_topic = 0*np.ones([docs_num, topic_num])
    topic_word_list = 0*np.ones([docs_num, topic_num, words_num])
    words_co_topic_list =  0*np.ones([topic_num, words_num, words_num])
    initialize_distributions()
    initial_docs_list()
    initialize_values_docs_list()
    compute_doc_topic()
    compute_topic_word()
    for i in range(0, docs_num):
        compute_topic_word_list_doc(i)
    print("initialization finished")
    return

def run(t_data, start, end_iter, iterations, save_p, clip, c_len, palpha, pbeta, pgamma):
    global topic_num, iteration_num, data_clip, data, docs_num, topic_num, words_num, context_len, alpha, beta, gamma
    data=t_data
    alpha = palpha
    beta = pbeta
    gamma = pgamma  
    context_len = c_len
    save_path = save_p
    data_clip = clip
    topic_num = start
    iteration_num = iterations
    
    create_dictionary(data) 
    docs_num = len(data)
    topic_num = start
    words_num = len(word_index)
    for i in range(0, end_iter):
        initialize()
        parameter_estimation()
        save_result(save_path)
        topic_num += 2
        np.save("LDA_runtime_"+str(data_clip)+".npy", total_time)
    return 
