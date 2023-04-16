#!/bin/bash

graphDataDir="../data/graphData/"
printf "cid:ID(Concept)\t:LABEL\tcluster\n" > $graphDataDir/concepts_header.tsv
printf ":START_ID(Concept)\trel\t:END_ID(Concept)\tp\n" > $graphDataDir/triples_header.tsv
printf "pid:ID(Paper)\tpaper\n" > $graphDataDir/papers_header.tsv