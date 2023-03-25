---
title: "Performant Text Classification with Naive Bayes for the Kaqchikel Mayan language"
date: "March 2023"
author: "William J. Wakefield"
github: https://github.com/Chok-Ketzamtzib/18337-project-kaqchikel-NLP
---

# Abstract

Kaqchikel is a language in the Mayan language family, spoken in Guatemala by about 500,000 people. As of this writing, there are no existing Natural Language Processing (NLP) application for this Mayan language, which leads to a lack of tools that could be implemented to help preserve the language in an advancing and globalized world. The text classification will use a Term Frequency- Inverse Document Frequency (TF-IDF) parallel model and a parallel Naive Bayes algorithm to perform sentiment analysis on the Kaqchikel Chronicles, a collection or rare pre-colonial texts. 

# Diagram

![It's Pizza](https://github.com/JacksonBurns/18337-project-template/blob/main/paper/images/pizza.png?raw=true)

# Algorithm

$$P(c|x) = P(x|c) * P(c) / P(x)$$

# References

Julia implementation for Naive Bayes https://juliatext.github.io/TextAnalysis.jl/latest/classify/

Julia Machine Learning Framework https://github.com/JuliaAI/MLJText.jl 

Houda Amazal, Mohammed Ramdani, and Mohamed Kissi. 2018. A Text Classification Approach using Parallel Naive Bayes in Big Data Context. In Proceedings of the 12th International Conference on Intelligent Systems: Theories and Applications (SITA'18). Association for Computing Machinery, New York, NY, USA, Article 36, 1–6. https://doi.org/10.1145/3289402.3289536

Annals of the Cakchiqueles https://www.gutenberg.org/files/20775/20775-h/20775-h.htm#THE_ANNALS

Maxwell, Judith M. and Robert M., II Hill. Kaqchikel Chronicles: The Definitive Edition. University of Texas Press, 2006. Project MUSE muse.jhu.edu/book/45051.