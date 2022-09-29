Data science & machine learning portfolio comprising projects from 2021/2022.
Click on the project titles to be redirected to the respective GitHub repositories.

# [Project 1: A Comparison of Strategies for Detecting Biased Text](https://github.com/joanrossello/Bias-Text-Detection)
This project investigates different strategies for detecting bias text in Natural Language Processing. In particular:

* Use data processing to tokenize text datasets, produce Part-Of-Speech tagging and dependency tagging, and prepare the dataloaders for the bias detection models.
* Explore neural network classifiers with GloVe word embeddings, and features relating to syntactic and sematic relationships of the word under analysis and its context.
* Train and test LSTMs BiLSTMS and transformers for text classification.
* Fine-tune pre-trained large language models such as BERT and BLOOM for bias text detection.
* Develop graphical models for text classification. This involves processing the text data to generate the graph network it represents through PMI and TF-IDF scores of corpus documents and unique words, and generate the classification model by training graph convolutional networks, both transductive and inductive.
* Show that pre-trained language models display high performance and generalisation in detecting different types of bias, especially when combined with bias-related word embeddings, and bidirectional LSTMs (mixed endemble architectures).


# [Project 2: Image Captioning with Transformers](https://github.com/joanrossello/Image-Captioning)
Image captioning is a now well-established challenge in the Computer Vision/Natural Language Processing community, which consists in generating an accurate description, or caption, of a given input image. This is typically done using a Convolutional Neural Network (CNN) to extract features from the image followed by a language model to sequentially predict an output sentence from these features. This work aims to investigate if by simply using the current state-of-the-art **transformer** architectures for feature extraction and sequence processing, we can train an accurate image captioning network with a limited amount of resources. We also investigate if incorporating the word embeddings and bounding boxes of objects detected in the image in a **multi-task learning** approach, can further improve the model’s accuracy. The models are evaluated on the Flickr8k and Flickr30k datasets using the BLEU metric with beam search.

![](/Images/img1_1.png) ![](/Images/img1_2.png)

The images show example Predictions (P) with 5 potential targets (T1-T5) on the test set with highest BLEU score.


# [Project 3: Information Retrieval Models for Passage Re-ranking](https://github.com/joanrossello/Information-Retrieval-Models)
Develop information retrieval models that solve the problem of passage retrieval, i.e. given a query, return a ranked list of short texts (passages).

* Text statistics: extract 1-grams from raw text and perform text pre-processing steps (expand contractions, remove punctuation and stop words, tokenization, lemmatization, etc.). Count word occurrences and analysie Zipf's distribution.
* Generate an inverted index of the unique terms in the corpus and their occurrences in each passage and query.
* Use TF-IDF vector representation of passages, cosine similarity and BM25 score to extract top 100 passages for each query in ranking order. 
* Implement query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing.
* Evaluate retrieval quality by computing average precision and NDCG metrics.
* Create feature representation for each query and passage pair, such as GloVe word embeddings, cosine similarity, sequence length, etc., and implement a logistic regression model to assess relevance of a passage to a given query.
* Use the LambdaMART learning to rank algorithm to learn a model that can re-rank passages.
* Build a neural network based model with Pytorch that can re-rank passages.


# [Project 4: Multi-task Learning Approaches for Animal Segmentation](https://github.com/joanrossello/Multitask-Image-Segmentation)




# [Project 5: ]()



# [Project 6: ]()



# [Project 7: ]()



# [Project 8: ]()

