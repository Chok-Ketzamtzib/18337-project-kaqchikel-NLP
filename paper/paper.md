---
title: "Performant Text Classification with Naive Bayes for the Kaqchikel Mayan language"
date: "March 2023"
author: "William J. Wakefield"
github: https://github.com/Chok-Ketzamtzib/18337-project-kaqchikel-NLP
---

# Abstract

Kaqchikel is a language in the Mayan language family, spoken in Guatemala by about 500,000 people. As of this writing, there are no existing Natural Language Processing (NLP) application for this Mayan language, which leads to a lack of tools that could be implemented to help preserve the language in an advancing and globalized world. The text classification will use a Term Frequency- Inverse Document Frequency (TF-IDF) parallel model and a parallel Naive Bayes algorithm to be evaluated on the Kaqchikel Chronicles, a collection or rare pre-colonial texts. The NLP pipeline developed may make contributions to TextAnalysis.jl, MLJ.jl, and within the Julia NLP ecosystem in general. 

# NLP Pipeline

![NLP Pipeline](https://github.com/JacksonBurns/18337-project-template/blob/main/paper/images/pipeline.png?raw=true)

# Plan

1. Create a Kaqchikel Database from dictionaries or existing databases
2. Perform text preprocessing by removing cases, numbers, HTML tags and punctuation (except glottals?)
3. Create the TF-IDF matrix with default Julia boilerplate
3. Create the Corpus object from the matrix
4. Pass Corpus object to default Naive Bayes Classifier
5. Evaluate the Pipeline
6. Parallelize and optimize the TF-IDF matrix portion in terms of feature extraction [5]
7. Parallelize and optimize the TF-IDF matrix portion in terms of feature extraction [5]
8. Compare results with the new pipeline with previous pipeline

# Parallel Naive Bayes Classifier

$$P(c|x) = P(x|c) * P(c) / P(x)$$

# References

[1] TextAnalysis.jl documentation by Julia Hub — https://docs.juliahub.com/TextAnalysis/5Mwet/0.7.3/

[2] Vectorize everything with Julia by Bence Komarniczky — https://towardsdatascience.com/vectorize-everything-with-julia-ad04a1696944

[3] MLJ framework — a machine learning framework of Julia: https://alan-turing-institute.github.io/MLJ.jl/dev/

[4] MLJ Data Interpretation and Scitypes: https://juliaai.github.io/DataScienceTutorials.jl/data/scitype/

[5] Houda Amazal, Mohammed Ramdani, and Mohamed Kissi. 2018. A Text Classification Approach using Parallel Naive Bayes in Big Data Context. In Proceedings of the 12th International Conference on Intelligent Systems: Theories and Applications (SITA'18). Association for Computing Machinery, New York, NY, USA, Article 36, 1–6. https://doi.org/10.1145/3289402.3289536

[6] Annals of the Cakchiqueles https://www.gutenberg.org/files/20775/20775-h/20775-h.htm#THE_ANNALS

[7] Maxwell, Judith M. and Robert M., II Hill. Kaqchikel Chronicles: The Definitive Edition. University of Texas Press, 2006. Project MUSE muse.jhu.edu/book/45051.

[8] Naive Bayes Example https://www.geeksforgeeks.org/applying-multinomial-naive-bayes-to-nlp-problems/

[9] [A corpus of K’iche’ annotated for morphosyntactic structure](https://aclanthology.org/2021.americasnlp-1.2) (Tyers & Henderson, AmericasNLP 2021)

[10] Mapreduce function in Julia to parallelize the preprocessing and naive bayes classifier (https://docs.julialang.org/en/v1/base/collections/#Base.mapreduce-Tuple{Any,%20Any,%20Any})