import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import time
import re
from PEGenerator import *


FLAG_CTR = 1



def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))

anchor = trans2tsp('10/10/2019 11:59:59 PM')
def parse_time_bucket(date):
    tsp = trans2tsp(date)
    tsp = tsp - anchor
    tsp = tsp//60
    return tsp

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def my_auc(label,score):
    false_score = score[label==0]
    positive_score = score[label==1]
    num_positive = (label==1).sum()
    num_negative = (label==0).sum()
    if num_positive==0:
        return 0.75
    if num_negative==0:
        return 1
    positive_score = positive_score.reshape((num_positive,1))
    positive_score = np.repeat(positive_score,num_negative,axis=1)
    false_score = false_score.reshape((1,num_negative))
    false_score = np.repeat(false_score,num_positive,axis=0)
    return 1-((positive_score<false_score).mean()+0.5*(positive_score==false_score).mean())

def evaluate_by_popularity(News,Impressions,ctr_flag = 1):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids)
        labels = Impressions[i]['labels']
        labels = np.array(labels)
        bucket = Impressions[i]['tsp']
        score = fetch_ctr_dim1(News,docids,bucket,ctr_flag)

        auc = my_auc(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    return AUC, MRR, nDCG5, nDCG10


def evaluate(user_scorings,news_scorings,Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids)
        labels = Impressions[i]['labels']
        labels = np.array(labels)
        uv = user_scorings[i]
        
        nv = news_scorings[docids]
        score = np.dot(nv,uv)
        

        auc = my_auc(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    return AUC, MRR, nDCG5, nDCG10

def evaluate_performance(rankings,Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(Impressions)):
        labels = Impressions[i]['labels']
        labels = np.array(labels)
        
        score = rankings[i]
        
        auc = my_auc(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    return AUC, MRR, nDCG5, nDCG10

def evaluate_cold_users(rankings,Impressions,test_click,num):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 =[]
    for i in range(len(Impressions)):
        labels = Impressions[i]['labels']
        labels = np.array(labels)
        
        uc = test_click[i]
        uc = (uc>0).sum()
        if not uc == num:
            continue
        
        score = rankings[i]
        
        auc = my_auc(labels,score)
        mrr = mrr_score(labels,score)
        ndcg5 = ndcg_score(labels,score,k=5)
        ndcg10 = ndcg_score(labels,score,k=10)
    
        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)
    
    AUC = AUC.mean()
    MRR = MRR.mean()
    nDCG5 = nDCG5.mean()
    nDCG10 = nDCG10.mean()
    
    return AUC, MRR, nDCG5, nDCG10


def ILAD(vecs):
    score = np.dot(vecs,vecs.T)
    score = (score+1)/2
    score = score.mean()-1/score.shape[0]
    score = float(score)
    return score

def ILMD(vecs):
    score = np.dot(vecs,vecs.T)
    score = (score+1)/2
    score = score.min()
    score = float(score)
    return score

def evaluate_density_ILxD(topk,rankings,Impressions,news_scoring):
    ILADs = []
    ILMDs = []
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids,dtype='int32')
        
        nv = news_scoring[docids]
        
        score = rankings[i]
        
        top_docids = score.argsort()[-topk:]
        
        nv = nv/np.sqrt(np.square(nv).sum(axis=-1)).reshape((nv.shape[0],1))
        
        nv = nv[top_docids]
        ilad = ILAD(nv)
        ilmd = ILMD(nv)

        ILADs.append(ilad)
        ILMDs.append(ilmd)

    ILADs = np.array(ILADs).mean()
    ILMDs = np.array(ILMDs).mean()
    return ILADs, ILMDs

def evaluate_diversity_topic_norm(topk,rankings,Impressions,News,Users,Category):
    topics = []
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids)
        score = rankings[i]
        top_args = score.argsort()[-topk:]
        top_docids = docids[top_args]
        verts = Category[top_docids].tolist()
        #print(top_args)
        label = Impressions[i]['labels']
        label = np.array(label)
        mask = label[top_args]
        
        verts = verts * mask
        
        verts = np.array(verts,dtype='int32')
        
        uc = Users.click[i]
        uverts = Category[uc].tolist()
        uverts = set(uverts) -{0}
        
        s = 0
        for v in verts:
            if v == 0:
                continue
            if not v in uverts:
                s += 1
        s /= (mask.sum()+0.01)
        topics.append(s)
    topics = np.array(topics).mean()
    return topics





def evaluate_diversity_topic_all(TOP_DIVERSITY_NUM,rankings,test_impressions,News,TestUsers):
    
    g3 = evaluate_diversity_topic_norm(TOP_DIVERSITY_NUM,rankings,test_impressions,News,TestUsers,News.vert)
    g4 = evaluate_diversity_topic_norm(TOP_DIVERSITY_NUM,rankings,test_impressions,News,TestUsers,News.subvert)
    

    metric = {
        'vert_norm_acc':g3,
        'subvert_norm_acc':g4,
    }
    
    return metric

import json
def dump_result(config,performance,cold,diversity):
    config = json.dumps(config)
    performance = json.dumps(performance)
    cold = json.dumps(cold)
    diversity = json.dumps(diversity)
    s = '\t'.join([config,performance,cold,diversity])
    s = s+'\n'
    with open('result.txt','a') as f:
        f.write(s)

def rel_news_ranking(user_scorings,news_scorings,Impressions):
    rankings = []
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids)
        
        uv = user_scorings[i]
        nv = news_scorings[docids]
        rel_score = np.dot(nv,uv)

        score = rel_score

        rankings.append(score)
    
    return rankings

def news_ranking(ranking_config,ctr_weight,activity_weights,user_scorings,news_scorings,
                 content_bias_score,content_bias_vecs,time_embedding_matrix,bias_content_scorer,
                 News,Impressions):
    rankings = []
    for i in range(len(Impressions)):
        docids = Impressions[i]['docs']
        docids = np.array(docids)
        bucket = Impressions[i]['tsp']
        
        publish_time = News.news_publish_bucket2[docids]

        if ranking_config['rel']:
            uv = user_scorings[i]
            nv = news_scorings[docids]
            rel_score = np.dot(nv,uv)
        else:
            rel_score = 0
        
        if ranking_config['content'] and not ranking_config['rece_emb']:
            bias_score = content_bias_score[docids]
        elif ranking_config['content'] and ranking_config['rece_emb']:
            bias_vecs = content_bias_vecs[docids]
            publish_time = bucket - publish_time
            arg = publish_time<0
            publish_time[arg] = 0
            publish_bucket = compute_Q_publish(publish_time)
            time_emb = time_embedding_matrix[publish_bucket]
            bias_vecs = np.concatenate([bias_vecs,time_emb], axis=-1)
            bias_score = bias_content_scorer.predict(bias_vecs)
            bias_score = bias_score[:,0]
        else:
            bias_score = 0
        
        if ranking_config['activity']:
            gate = activity_weights[i]
        else:
            gate = 0.5
        
        if ranking_config['ctr']:
            ctr = fetch_ctr_dim1(News,docids,bucket,FLAG_CTR)
        else:
            ctr = 0
        
        score = gate*rel_score + (1-gate)*(ctr*ctr_weight + bias_score)

        rankings.append(score)
    
    return rankings