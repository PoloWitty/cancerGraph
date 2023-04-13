#!/bin/bash

# generate triples
python get_triples.py
python drop_duplicate_triple.py

# generate concepts
python get_concept.py

# generate corresponding paper file
python get_papers.py