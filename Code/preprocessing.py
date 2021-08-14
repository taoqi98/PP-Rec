import numpy as np
import random
import os

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def get_train_input(session,news_index,config):
    npratio = config['npratio']
    
    sess_pos = []
    sess_neg = []
    user_id = []
    sess_buckets = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _,_,bucket, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            sess_buckets.append(bucket)
            user_id.append(sess_id)
    print(len(user_id))
    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    sess_buckets = np.array(sess_buckets)
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        #index = np.random.randint(1+npratio)
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, sess_buckets,user_id, label

def get_test_input(session,news_index):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _,_,tsp, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[],
                'tsp':tsp}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word