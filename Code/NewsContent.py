import numpy as np
from utils import *
import os
import json

class NewsContent():
    def __init__(self,config,):
        self.config = config
        self.read_news()
        self.get_doc_input()
        self.load_entitiy()
        self.load_ctr()
        self.load_publish_time()

    def fetch_news(self,docids,):
        title = None
        vert = None
        subvert = None
        body = None
        entity = None
        config = self.config
        if 'title' in config['attrs']:   
            title = self.title[docids]
        if 'vert' in  config['attrs']:
            vert = self.vert[docids]
            vert = vert.reshape(list(docids.shape)+[1])
        if 'subvert' in  config['attrs']:
            subvert = self.subvert[docids]
            subvert = subvert.reshape(list(docids.shape)+[1])
        if 'body' in config['attrs']:
            body = self.body[docids]
        if 'entity' in config['attrs']:
            entity = self.news_entity_index[docids]
            
        FeatureTable = {'title':title,'vert':vert,'subvert':subvert,'body':body,'entity':entity}
        feature = [FeatureTable[v] for v in config['attrs']]
        feature = np.concatenate(feature, axis=-1)
        return feature

    def read_news(self,):
        config = self.config
        news={}
        category=[]
        subcategory=[]
        news_index={}
        index=1
        word_dict={}
        with open(self.config['data_root_path']+'/docs.tsv') as f:
            lines=f.readlines()
        for line in lines:
            splited = line.strip('\n').split('\t')
            doc_id,vert,subvert,title= splited[0:4]
            if len(splited)>4:
                body = splited[-1]
            else:
                body = ''

            if doc_id in news_index:
                continue
            news_index[doc_id]=index
            index+=1
            category.append(vert)
            subcategory.append(subvert)
            title = title.lower()
            title=word_tokenize(title)
            for word in title:
                if not(word in word_dict):
                    word_dict[word]=0
                word_dict[word] += 1

            if 'body' in config['attrs']:
                body = body.lower()
                body = word_tokenize(body)
                for word in body:
                    if not(word in word_dict):
                        word_dict[word]=0
                    word_dict[word] += 1
                news[doc_id]=[vert,subvert,title,body]
            else:
                news[doc_id]=[vert,subvert,title,None]

        category=list(set(category))
        subcategory=list(set(subcategory))
        category_dict={}
        index=0
        for c in category:
            category_dict[c]=index
            index+=1
        subcategory_dict={}
        index=0
        for c in subcategory:
            subcategory_dict[c]=index
            index+=1
        word_dict_true = {}
        word_index = 1
        for word in word_dict:
            if word_dict[word]<config['word_filter']:
                continue
            if not word in word_dict_true:
                word_dict_true[word] = word_index
                word_index +=1
            
        self.news = news
        self.news_index = news_index
        self.category_dict = category_dict
        self.subcategory_dict = subcategory_dict
        self.word_dict = word_dict_true

    def get_doc_input(self):
        config = self.config
        news = self.news
        news_index = self.news_index
        category = self.category_dict
        subcategory = self.subcategory_dict
        word_dict = self.word_dict

        title_length = config['title_length']
        body_length = config['body_length']

        news_num=len(news)+1
        news_title=np.zeros((news_num,title_length),dtype='int32')
        news_vert=np.zeros((news_num,),dtype='int32')
        news_subvert=np.zeros((news_num,),dtype='int32')
        if 'body' in config['attrs']:
            news_body=np.zeros((news_num,body_length),dtype='int32')
        else:
            news_body = None
        for key in news:    
            vert,subvert,title,body=news[key]
            doc_index=news_index[key]
            news_vert[doc_index]=category[vert]
            news_subvert[doc_index]=subcategory[subvert]
            for word_id in range(min(title_length,len(title))):
                if title[word_id].lower() in word_dict:
                    word_index = word_dict[title[word_id].lower()]
                else:
                    word_index = 0
                news_title[doc_index,word_id]=word_index
            if 'body' in config['attrs']:
                for word_id in range(min(body_length,len(body))):
                    word = body[word_id].lower()
                    if word in word_dict:
                        word_index = word_dict[word]
                    else:
                        word_index = 0
                    news_body[doc_index,word_id]=word_index  

        self.title = news_title
        self.vert = news_vert
        self.subvert = news_subvert
        self.body = news_body
        
    def load_entitiy(self,):
        config = self.config
        news_index = self.news_index
        max_entity_num = config['max_entity_num']
        KG_root_path = config['KG_root_path']

        with open(os.path.join(KG_root_path,'entity2id.txt')) as f:
            lines = f.readlines()

        EntityId2Index = {}
        EntityIndex2Id = {}
        for i in range(1,len(lines)):
            eid, eindex = lines[i].strip('\n').split('\t')
            EntityId2Index[eid] = int(eindex)
            EntityIndex2Id[int(eindex)] = eid

        with open(os.path.join(KG_root_path,'V21UrlDocs22_title_josn.tsv')) as f:
            lines = f.readlines()

        news_entity = {}
        retain_entities = {}
        index = 1
        g = []
        for i in range(len(lines)):
            d = json.loads(lines[i].strip('\n'))
            docid = d['doc_id']
            if not docid in news_index:
                continue
            news_entity[docid] = []
            entities = d['entities']
            for j in range(len(entities)):
                e = entities[j]['Label']
                eid = entities[j]['WikidataId']
                if not eid in EntityId2Index:
                    continue
                if not eid in retain_entities:
                    retain_entities[eid] = index
                    index += 1
                news_entity[docid].append([e,eid])

        entity_embedding = np.zeros((len(retain_entities)+1,100))

        temp_entity_embedding = np.load(os.path.join(KG_root_path,'entity_embedding.npy'))

        for eid in retain_entities:
            retain_index = retain_entities[eid]
            index = EntityId2Index[eid]
            entity_embedding[retain_index,:] = temp_entity_embedding[index,:]

        news_entity_index = np.zeros((len(news_index)+1,max_entity_num),dtype='int32')

        for newsid in news_index:
            index = news_index[newsid]
            entities = news_entity[newsid]
            ri = np.random.permutation(len(entities))
            for j in range(min(len(entities),max_entity_num)):
                eid = entities[ri[j]][-1]
                news_entity_index[index,j] = retain_entities[eid]

        self.entity_embedding = entity_embedding
        self.news_entity_index = news_entity_index
        self.news_entity = news_entity
        self.retain_entities = retain_entities
        
    def load_ctr(self,):
        news_index = self.news_index
        popularity_path = self.config['popularity_path']
        with open(os.path.join(popularity_path,'mergeimps.tsv')) as f:
            lines = f.readlines()

        num = 1100
        news_stat_imp = np.zeros((len(news_index)+1,num))
        news_stat_click = np.zeros((len(news_index)+1,num))
        mx = -1
        for i in range(len(lines)):
            newsid, bucket, click, imp = lines[i].strip('\n').split('\t')
            if not newsid in news_index:
                continue
            nindex = news_index[newsid]
            bucket = int(bucket)
            click = int(click)
            imp = int(imp)
            news_stat_imp[nindex,bucket] += imp
            news_stat_click[nindex,bucket] += click
            if bucket>mx:
                mx = bucket

        self.news_stat_imp = news_stat_imp
        self.news_stat_click = news_stat_click

    def load_publish_time(self,):
        news_index = self.news_index
        popularity_path = self.config['popularity_path']

        news_publish_bucket2 = np.zeros((len(news_index)+1,),dtype='int32')
        with open(os.path.join(popularity_path,'docs_pub_time.tsv')) as f:
            lines = f.readlines()

        for i in range(len(lines)):
            nid, tsp = lines[i].strip('\n').split('\t')
            if not nid in news_index:
                continue
            nindex = news_index[nid]
            if tsp=='':
                tsp = '10/10/2019 11:59:59 PM'
            #tsp = trans2tsp(tsp)
            bucket = parse_time_bucket(tsp)
            news_publish_bucket2[nindex,] = bucket

        index = news_publish_bucket2<0
        news_publish_bucket2[index] = 0
        self.news_publish_bucket2 = news_publish_bucket2

        news_publish_bucket = np.zeros((len(news_index)+1,),dtype='int32')
        news_stat_imp = self.news_stat_imp
        for i in range(1,news_stat_imp.shape[0]):
            start = (news_stat_imp[i]>0).argmax()
            news_publish_bucket[i,] = start
        self.news_publish_bucket = news_publish_bucket