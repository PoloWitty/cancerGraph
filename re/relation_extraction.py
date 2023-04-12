"""
desc:	do relation extraction using OpenNRE
author:	Yangzhe Peng
date:	2023/03/28
"""

import typer
import json
import tqdm
from pathlib import Path
from typing import Optional, List, Tuple,Set
import subprocess
import opennre
import pdb
from itertools import permutations
from rich import print
from tqdm import trange
import torch

def get_line_num(filename):
    out = subprocess.getoutput(f'wc -l {filename}|cut -d " " -f1')
    return int(out)

# https://github.com/allenai/allennlp/blob/main/allennlp/data/dataset_readers/dataset_utils/span_utils.py
# ### Start Code
def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str,
) -> bool:
    if curr_bio_tag == "B":
        return True
    if curr_bio_tag == "I" and prev_bio_tag == "O":
        return True
    if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
        return True
    return False

def iob1_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to IOB1 tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
    by "I-LABEL" or "B-LABEL").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    prev_bio_tag = None
    prev_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]

        if curr_bio_tag not in ["B", "I", "O"]:
            raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
        if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            # bio_tag == "I" and curr_conll_tag == active_conll_tag
            # We're continuing a span.
            span_end += 1

        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)
# ### end code


def main(
        nerResult_filename:Path = Path('../data/textData/ner/biobert.jsonl'),
        output_filename:Path = Path('../data/textData/re/data.jsonl'),
        openNRE_model:str = 'wiki80_bert_softmax',
        use_cuda:bool = True,
        pred_threshold:float = 0.96,
        position:int = 0
):
    """
        will read `biobert.jsonl`. Enum all the entity pair in the abstract and keep those predict possibility greater than the threshold
        param:
            nerResult_filename
            output_filename_prefix
            openNRE_model: possible values: ['wiki80_bert_softmax','wiki80_bertentity_softmax','tacred_bert_softmax','tacred_bertentity_softmax']
            use_cuda: whethre use cuda for openNRE model
            pred_threshold: range(0,1)
            position: split idx

    """
    assert openNRE_model in ['wiki80_bert_softmax','wiki80_bertentity_softmax','tacred_bert_softmax','tacred_bertentity_softmax'], 'OpenNRE framework do not support this model name'
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    model = opennre.get_model(openNRE_model)
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    with open(nerResult_filename) as fp:
        with trange(get_line_num(nerResult_filename),desc='process'+str(position),dynamic_ncols=True,position=position) as pbar:
            for _ in pbar:
                line = fp.readline().strip()
                obj = json.loads(line)

                # get char position for all the words
                charSum = 0
                offset_word2char = {}
                for i,w in enumerate(obj['abstract']):
                    s = charSum
                    e = charSum + len(w)
                    charSum += len(w)+1
                    offset_word2char[i] = (s,e)

                # decode the token level entity position and convert it to char level
                spans = iob1_tags_to_spans(obj['ner_abstract'])
                entity_charPos = []
                for (tag,(start,end)) in spans:
                    s = offset_word2char[start][0]
                    e = offset_word2char[end][1]
                    entity_charPos.append([[s,e],tag])

                # get relation extraction result
                text = ' '.join(obj['abstract'])
                obj['re'] = []
                for i,(h,tag_h) in enumerate(entity_charPos):
                    for j,(t,tag_t) in enumerate(entity_charPos):
                        if j<=i-10 or j>=i+10:
                            continue
                        if i==j:
                            continue
                        if tag_h==tag_t:
                            continue
                        res = model.infer({'text': text, 'h': {'pos': h}, 't': {'pos': t}})
                        if res[1]>pred_threshold:
                            # print(f'{text[h[0]:h[1]]}\t{res}\t{text[t[0]:t[1]]}')
                            obj['re'].append([[spans[i][1],res[0],spans[j][1]],res[1]])
                output_fp.write(json.dumps(obj)+'\n')
            pbar.close()
        output_fp.close()

if __name__=='__main__':
    typer.run(main)
