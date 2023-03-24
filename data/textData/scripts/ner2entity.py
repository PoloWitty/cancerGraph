"""
desc:	convert ner result to 'entityName, abs_idx, word_id, type'format
author:	Yangzhe Peng
date:	2023/03/22
"""
import tqdm
import json
from typing import Optional,List,Union,Tuple,Set
import pdb
import subprocess
from collections import defaultdict

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

if __name__=='__main__':
    ner_data_filename = 'ner/biobert.jsonl'
    output_filename = 'ner/entities.tsv'
    output_entity2idx_filename = 'ner/entity2idx.json'

    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    
    idx = 0
    entity2idx = defaultdict(list)
    with open(ner_data_filename) as fp:
        for abs_idx in tqdm.trange(get_line_num(ner_data_filename),desc='processing'):
            line = fp.readline().strip()
            obj = json.loads(line)

            tokens = obj['abstract'];ner_tags = obj['ner_abstract']
            spans = iob1_tags_to_spans(ner_tags)
            
            for (tag,(s,e)) in spans:
                text = ' '.join(tokens[s:e+1])
                entity = {
                    'text':text,
                    'abs_idx':abs_idx,
                    'start_idx': s,
                    'tag': tag
                }
                output_fp.write(f"{entity['text']}\t{entity['abs_idx']}\t{entity['start_idx']}\t{entity['tag']}\n")
                idx += 1
                entity2idx[text].append(idx)

    print('dumping entity2idx...')
    json.dump(entity2idx,open(output_entity2idx_filename,'w'),indent=4)

    # entity2idx = json.load(open(output_entity2idx_filename))
    uniqentities_filename = ''.join(output_filename.split('/')[:-1])+'/uniq_entities.txt'
    with open(uniqentities_filename,'w') as fp:
        for e in tqdm.tqdm(entity2idx.keys(),desc='writing uniq entities to '+uniqentities_filename):
            fp.write(e+'\n')