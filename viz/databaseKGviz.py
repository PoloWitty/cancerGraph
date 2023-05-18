'''
desc:   use streamlit to viz the graph
author: Yangzhe Peng
date:   2023/4/16
'''

import streamlit as st
from neo4j import GraphDatabase
from streamlit_echarts import st_echarts
from pytz import timezone
import datetime
import pandas as pd
from itertools import permutations
import json
import pdb
from pathlib import Path
from tqdm.rich import tqdm
from typing import List,Set

@st.cache_resource
def get_database_driver():
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "admin"))
    return driver

@st.cache_resource
def get_entity2concept(concept2entity_filename:Path = Path('../data/textData/cluster/concept2entity.json')):
    with open(concept2entity_filename) as fp:
        print('reading concept2entity...')
        concept2entity = json.load(fp)
    
    entity2concept = {}
    for c,entities in tqdm(concept2entity.items(),desc='generating entity2concept map'):
        for entity in entities:
            entity2concept[entity] = c
        if entity2concept.get(c,'')=='':
            entity2concept[c] = c
        
    return entity2concept

def query(tx, cid,limitNum=10):
    result = tx.run('\
    MATCH (h:Concept { cid:$cid })-[r]->(t:Concept)\
    RETURN r,t\
    limit '+str(limitNum)+';\
    ', cid = cid )
    return [record for record in result]

def queryReserve(tx,cid,limitNum=10):
    result = tx.run('\
    MATCH (h:Concept)-[r]->(t:Concept {cid:$cid})\
    RETURN h,r\
    limit '+str(limitNum)+';\
    ', cid = cid )
    return [record for record in result]

def queryCidDetail(tx,cid):
    result = tx.run('''
        MATCH(n:Concept {cid:$cid})
        RETURN n
    ''',cid=cid)
    return result.single()

def queryPaper(tx,pid):
    result = tx.run('''
        MATCH (p:Paper {pid:$pid})
        RETURN p
        ''',pid=pid)
    return result.single()

def queryEach(tx,srcId,tgtId):
    result = tx.run('''
    MATCH (a:Concept {cid:$srcId})-[r]->(b:Concept {cid:$tgtId})
    RETURN r
    ''',srcId=srcId,tgtId=tgtId)
    return [record for record in result]

@st.cache_data(persist='disk')
def get_mentionedPaper(pids:List[str]):
    global driver
    papers = []
    with driver.session() as session:
        for pid in pids:
            record = session.execute_read(queryPaper,pid)
            paper = json.loads(record['p']._properties['paper'])
            papers.append(paper)
    return papers

@st.cache_data(persist='disk')
def get_result(queryName,LIMIT_NUM):
    global driver,entity2concept,legend
    
    queryCid = entity2concept.get(queryName,'')
    if queryCid=='':
        return [None]*5 # 5 is the num of regular return values
    
    with driver.session() as session:
        links = []
        nodes = []
        nodeProp = {}
        srctgt2rel = {}
        nodeSet = set()

        # query center node
        record = session.execute_read(queryCidDetail,queryCid)
        type_ = [l for l in record['n'].labels if l!='Concept'][0]
        nodeProp[queryCid]={
            'name':queryCid,
            'type': type_,
            'cluster':json.loads(record['n']._properties.get('cluster'))
        }
        
        # query node and relation link from center node
        result = session.execute_read(query,queryCid,LIMIT_NUM)
        for record in result:
            cid  = record['t']._properties['cid']
            type_ = [t for t in record['t'].labels] # type
            type_.pop(type_.index('Concept'))
            nodeProp[cid] = {
                "name": cid,
                "type": type_[0],
                "cluster": json.loads(record['t']._properties.get('cluster'))
            }
            
            srctgt2rel[(queryCid,cid)] = (record['r'].get('rel'),record['r'].get('p'))

        # # query relation between 1-hop nodes(need to follow node2center because this use previous result)
        # perms = permutations(range(min(LIMIT_NUM,len(result))),2)
        # for perm in perms:
        #     src = perm[0]
        #     tgt = perm[1]
        #     srcId = result[src]['t']._properties['cid']
        #     tgtId = result[tgt]['t']._properties['cid']
        #     if srcId==queryCid or tgtId==queryCid:
        #         continue
        #     res = session.execute_read(queryEach,srcId,tgtId)
        #     if len(res)!=0:
        #         for record in res:
        #             srctgt2rel[(srcId,tgtId)] = (record['r'].get('rel'),record['r'].get('p'))


        # query nodes and relations link to center node
        result = session.execute_read(queryReserve,queryCid,int(LIMIT_NUM//5))
        for record in result:
            cid  = record['h']._properties['cid']
            type_ = [t for t in record['h'].labels] # type
            type_.pop(type_.index('Concept'))
            nodeProp[cid] = {
                "name": cid,
                "type": type_[0],
                "cluster": json.loads(record['h']._properties.get('cluster'))
            }
            srctgt2rel[(cid,queryCid)] = (record['r'].get('rel'),record['r'].get('p'))
            
        # add some viz effect
        for (src,tgt),rel in srctgt2rel.items():
            if tgt not in nodeSet and tgt!=queryCid:
                category = legend.index(nodeProp[tgt]['type'])
                nodes.append({
                    'name':tgt,
                    'symbolSize':15,
                    # 'itemStyle':{
                    #     'color':'#9eb2d2'
                    # },
                    'value': nodeProp[tgt]['name'],
                    'category': category
                })
                nodeSet.add(tgt)
            if src not in nodeSet and src!=queryCid:
                category = legend.index(nodeProp[src]['type'])
                nodes.append({
                    'name':src,
                    'symbolSize':15,
                    # 'itemStyle':{
                    #     'color':'#9eb2d2'
                    # },
                    'value': nodeProp[src]['name'],
                    'category': category
                })
                nodeSet.add(src)
            if src == tgt:
                continue

            links.append({
                'source':src,
                'target':tgt,
                'value':srctgt2rel[(src,tgt)],
                'label':{
                    'formatter':'{@[0]}'
                }
            })
        
    # add center node
    nodes.append({
        'name':queryCid,
        'value': queryName,
        'symbolSize':20,
        'itemStyle':{
            'color':'#836fd9'
        },
    })
    return queryCid, nodes, links, nodeProp, srctgt2rel

def feedbackHandler(userInput:str):
    utcNow = datetime.datetime.now(timezone('utc'))
    shanghaiNow = utcNow.astimezone(timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S %Z%z")
    with open('./log/dbpediaVizFeedBack.log','at') as fp:
        fp.write(shanghaiNow+'\t'+userInput.replace('\t',' ')+'\t'+st.session_state['userFeedback'].replace('\t',' ')+'\n')

def dict_viz(prop:str):
    prop_dict = {}
    for k,v in eval(prop).items():
        if type(v)==list:
            prop_dict[k] = ', '.join(v)
        else:
            prop_dict[k] = v
    prop_table = pd.DataFrame(prop_dict,index=['value']).transpose()
    return prop_table

def centerButtonHandler(name):
    st.session_state['queryName'] = name
    
def construct_echart_settings(nodes:List,links:List):
    '''
    construct echart options from the result of get_result
    '''
    global legend
    options = {
        "tooltip": {},
        "animationDuration": 1500,
        "animationEasingUpdate": "quinticInOut",
        "legend":[
            {
                'data': legend,
                'orient':'vertical',
                'align':'left',
                'top':'top',
                'left':'left'
            }
        ],
        "series": [
            {
                "type": "graph",
                # layout
                "layout": "force",
                "force" : {
                    'initLayout':'circular',
                    'edgeLength': [30,150],
                    'repulsion': 150, 
                    'gravity': 0.3,
                    'friction':0.5
                },
                # node
                "label": {
                    "show":True,
                    "position": "right", 
                    "formatter": "{c}"
                },

                "categories":[
                    {'name':legend[0]},
                    {'name':legend[1]},
                    {'name':legend[2]},
                    {'name':legend[3]},
                ],
                
                # edge
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [2, 15],
                "edgeLabel": {
                    "show":True,
                    "fontSize": 10,
                    "formatter":'{c}',
                },
                "lineStyle": {
                    # "color": "#ecedea", 
                    "curveness": 0.2,
                    "width":5,
                    "opacity":0.9
                },
                
                # others
                "roam": True,
                "scaleLimit":{
                    'min':'2.5',
                    'max':'6'
                },
                "zoom": 3.25,
                "draggable":False,
                "emphasis": {
                    "focus": "adjacency", 
                    "lineStyle": {"width": 10}
                },

                
                # dataset
                "data": nodes,
                "links": links,
                }
            ]
        }
    return options


if __name__=='__main__':
    st.set_page_config(
        page_title="cancerKG:v0",
        layout="wide",
        # initial_sidebar_state="expanded",
    )
    LIMIT_NUM = 30
    driver = get_database_driver()
    entity2concept = get_entity2concept()
    legend = ['disease','gene','chemical','species']
    # st.cache_data.clear()

    col0,col1,col2 = st.columns([1.5,4,1.5])
    
    with col0:
        queryName = st.text_input('Search Entity name','cancer',key='queryName')
        queryCid, nodes,links,nodeProp,srctgt2rel = get_result(queryName,LIMIT_NUM)
    if queryCid==None:
        # no such entity
        with col1:
            st.error('No such Entity in the database.',icon='❌')
    else:
        
        with open('./linear_graph.log','a') as fp:
            tmp = ''
            for (src,tgt),rel in srctgt2rel.items():
                tmp += f'({src},{rel[0]},{tgt})'
            fp.write(f'{queryName}\n')
            fp.write(tmp+'\n')

        # get mentioned papers
        pids = nodeProp[queryCid]['cluster'].get(queryName,nodeProp[queryCid]['cluster'][queryCid])
        if len(pids)>=10:
            end = 10
        else:
            end = -1
        mentionedPapers = get_mentionedPaper(pids[:end])
        
        # echarts setting
        options = construct_echart_settings(nodes,links)
        with col1:
            event = {
                "click": "function(params){return params.data}", # minizie this use https://www.minifier.org/
            }
            clickEvent = st_echarts(options,events=event,height='500px')
            
            # viz mentioned papers
            with st.container():
                st.subheader('where is this entity mentioned')
                for paper in mentionedPapers:
                    with st.expander('Title: **'+' '.join(paper['title'])+'**'):
                        # del paper['title']
                        viz_abs = ''
                        for token in paper['abstract']:
                            if token.lower() != queryName.lower():
                                viz_abs+= token + ' '
                            else:
                                viz_abs+= ':red['+token+'] '
                                
                        # st.markdown('- author: '+paper['authors'])
                        # st.markdown('- journal: '+paper['journal'])
                        # st.markdown('- pubYear: '+paper['pubdate'])
                        st.markdown('- :green['+paper['authors']+' - '+paper['journal']+' - '+paper['pubdate']+']')
                        if paper['doi']!='':
                            st.markdown('- https://doi.org/'+paper['doi'])
                        st.markdown('- abstract: '+viz_abs)
                        # st.json(paper)
                if len(pids)>=10:
                    st.text('(More...)')
        with col0:
            st.subheader(queryName)
            st.caption('cid: '+queryCid)
            st.markdown('\[type\]: '+nodeProp[queryCid]['type'])
            alias_viz = lambda alias:', '.join(list(alias)[:30])+' (and more...)' if len(alias)>=50 else ', '.join(alias)
            st.markdown('\[alias\]: '+alias_viz(nodeProp[queryCid]['cluster'].keys()))
            
        with col2:
            with st.expander('Feedback'):
                with st.form('feedback'):
                    st.text_area('Feedback',key='userFeedback',label_visibility='collapsed',placeholder='Write your feedback here🤝')
                    submitted = st.form_submit_button("Submit",on_click=feedbackHandler,kwargs={'userInput':queryCid})
                    if submitted:
                        st.write('Thanks for your feedback! 🥰')
        
        # process the node click event
        try:
            if clickEvent.get('name','')!='':
                with col2:
                    cid = clickEvent['name']
                    st.subheader(cid) 
                    st.caption('cid: '+cid)
                    st.markdown('\[type\]: '+nodeProp[cid]['type'])
                    st.markdown('\[alias\]: '+alias_viz(nodeProp[cid]['cluster'].keys()))
                    st.button('click to display graph around '+cid,on_click=centerButtonHandler,kwargs={'name':cid})
        except AttributeError:
            pass