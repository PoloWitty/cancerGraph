import opennre
from rich import print
model = opennre.get_model('wiki80_bert_softmax')
# model = opennre.get_model('wiki80_bertentity_softmax')

data= {"title":["Machine","learning","to","predict","early","recurrence","after","oesophageal","cancer","surgery"],"abstract":["to","develop","a","predictive","model","for","early","recurrence","after","surgery","for","oesophageal","adenocarcinoma","using","a","large","multi - national","cohort",".","summary","background","dataearly","cancer","recurrence","after","oesophagectomy","is","a","common","problem","with","an","incidence","of","20 - 30","%","despite","the","widespread","use","of","neoadjuvant","treatment",".","quantification","of","this","risk","is","difficult","and","existing","models","perform","poorly",".","machine","learning","techniques","potentially","allow","more","accurate","prognostication","and","have","been","applied","in","this","study",".","methodsconsecutive","patients","who","underwent","oesophagectomy","for","adenocarcinoma","and","had","neoadjuvant","treatment","in","6","uk","and","1","dutch","oesophago - gastric","units","were","analysed",".","using","clinical","characteristics","and","post - operative","histopathology",",","models","were","generated","using","elastic","net","regression","(","elr",")","and","the","machine","learning","methods","random","forest","(","rf",")","and","xg","boost","(","xgb",")",".","finally",",","a","combined","(","ensemble",")","model","of","these","was","generated",".","the","relative","importance","of","factors","to","outcome","was","calculated","as","a","percentage","contribution","to","the","model",".","resultsin","total","812","patients","were","included",".","the","recurrence","rate","at","less","than","1","year","was","29. 1","%",".","all","of","the","models","demonstrated","good","discrimination",".","internally","validated","aucs","were","similar",",","with","the","ensemble","model","performing","best","(","elr = 0. 785",",","rf = 0. 789",",","xgb = 0. 794",",","ensemble = 0. 806",")",".","performance","was","similar","when","using","internal - external","validation","(","validation","across","sites",",","ensemble","auc = 0. 804",")",".","in","the","final","model","the","most","important","variables","were","number","of","positive","lymph","nodes","(","25. 7","%",")","and","vascular","invasion","(","16. 9","%",")",".","conclusionsthe","derived","model","using","machine","learning","approaches","and","an","international","dataset","provided","excellent","performance","in","quantifying","the","risk","of","early","recurrence","after","surgery","and","will","be","useful","in","prognostication","for","clinicians","and","patients",".","draft","visual","abstract","o _ fig","o _ linksmallfig","width = 200","height = 110","src =","' '","figdir \/ small \/ 19001073v1 _ ufig1. gif","' '","alt =","' '","figure","1","' '",">","view","larger","version","(","26k",")",":","org. highwire. dtl. dtlvardef","@","1d0693corg. highwire. dtl. dtlvardef","@","1ace"],"journal":"","authors":"Rahman, S.; Walker, R. C.; LLoyd, M. A.; Grace, B. L.; van Boxel, G. I.; Kingma, F. B.; Ruurda, J. P.; van Hillegersberg, R.; Harris, S.; Parsons, S.; Mercer, S.; Griffiths, E. A.; O'Neill, J. R.; Turkington, R.; Fitzgerald, R.; Underwood, T. J.; OCCAMS Consortium,  ","pubdate":"2019","doi":"10.1101\/19001073","src":"medrxiv","pmid":"",'ner':[[11,12],[22,22],[78,78],[242,242]],"ner_abstract":["O","O","O","O","O","O","O","O","O","O","O","B-disease","I-disease","O","O","O","O","O","O","O","O","O","B-disease","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-disease","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-disease","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]}


charSum = 0
offset_word2char = {}
for i,w in enumerate(data['abstract']):
    s = charSum
    e = charSum + len(w)
    charSum += len(w)+1
    offset_word2char[i] = (s,e)

entity_charPos = []
for entity in data['ner']:
    s = offset_word2char[entity[0]][0]
    e = offset_word2char[entity[1]][1]
    entity_charPos.append([s,e])

text = ' '.join(data['abstract'])
print(text)
for h in entity_charPos:
    for t in entity_charPos:
        res = model.infer({'text': text, 'h': {'pos': h}, 't': {'pos': t}})
        if res[1]>0.96:
            print(f'{text[h[0]:h[1]]}\t{res}\t{text[t[0]:t[1]]}')