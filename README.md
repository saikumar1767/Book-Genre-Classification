# Book-Genre-Classification

## Abstract

Book genre classification has the potential to revolutionize the publishing industry, making it easier for readers to discover books that match their interests and preferences. Our project outlines the attempt to predict how well the summary of a book description will be in determining the genre. For our project, we trained the following models:
LSTM, BERT (Google), BART (Facebook), and Google Pegasus. The findings imply that predicting a book’s genre strongly depends on how well the text has been described without losing factual information, along with other factors besides it.

## 1 Introduction

Books have had a tremendous impact on the globe for many years. They can affect society, politics, and laws. They can also influence how people think and behave. While natural language processing has previously been applied to books, predicting the genre based on the summary has yet to be done to this extent. This methodology enables users to predict the genre categorization conclusion by considering the book’s comprehensive description and summarizing the book description as an intermediary procedure. In this project, NLP assisted me in summarizing a book description’s essential features into a plot, which was then utilized to detect originality and other variables to aid in genre prediction success. I employed LSTM and BART (Abstractive summarization) to produce summaries from a given book description and then used Google’s BERT classification to predict the genres of each book. This project aims to simplify the task of book genre prediction for readers, making it easier for them to find books that match their interests. With so many books available, readers often struggle to find books that they will enjoy. This project reduces the guesswork and uncertainty involved in book selection by accurately predicting the genre of a book. This helps readers quickly identify books that match their preferences, saving them time and effort.

## 2 Related work

There are two main aspects to our project:
• Text Summarization of Book Description
• Prediction of Genre

### 2.1 Text Summarization

There are two types of text summarization: extractive and abstractive. Abstractive text summarization involves creating a summary that is not a verbatim copy of the original text but rather a new, shorter text that captures the critical information and meaning of the original text. I used Abstract summarization to summarize book descriptions and predict their genre more efficiently. The process involved generating new sentences that capture the essence of the original text of the book description more concisely. I then train a model on the summarizations to predict the genre. This approach reduces computational costs for pre-
dictating genre from lengthy book descriptions. My work differs from previous studies on text summarization for book descriptions. I performed text summarization using LSTM and used the resulting summarized descriptions for downstream tasks, specifically book genre prediction. To the best of my knowledge, previous work only focused on text summarization and did not utilize the summarizations for any further applications. Thus, my work represents a novel approach that goes beyond text summarization and demonstrates the potential of summarization for improving the efficiency of downstream tasks. My approach of predicting the genre from both the original long and summarized descriptions allows me to compare the accuracies of the two methods.

### 2.2 Prediction of Genre

Several scholars have considered using NLP to book description scripts to identify specific patterns and determine the genre. For example, Gupta et al. (2019) proposed that the method gains knowledge from many words from books and transforms them into a feature matrix. During transformation, the size of the initial matrix is reduced using Wordnet and Principle Component Analysis. Then, the AdaBoost classifier is applied to predict the genres of the books. Ozsarfati et al. (2019). Each algorithm was fine-tuned to attain the best parameter values, while no modifications were conducted on the dataset. Machine learning algorithms classify certain text data, resulting in better predictions and accuracy. Text classification techniques are analyzed and worked on to improve the accuracy of structured data. Moreover, we are fusing the ideas from the publications above to create characteristics that will aid in curating our analysis-based genre prediction.

## 3 Problem Description

In this project, I explored and compared the efficiency of two methods: predicting the genre from the original description, summarizing a long plot into a shorter text (summary), and then classifying the genre from the summary. This task is challenging because different genres have similar plot structures, and a book can sometimes belong to multiple genres. In natural language processing tasks such as text summarization, the length of the input text can significantly impact computational time. Longer input texts require more processing time to analyze and understand, which can result in slower performance and higher resource requirements. However, the output of a text summarization task, a condensed version of the original text, typically requires less computational time and resources to process and analyze. This is because the summary contains only the most essential and relevant information from the original text, which is typically much shorter in length.

## 4 Methods

### 4.1 Technical Procedure

This project mainly involves two steps, as shown in Figure 4.1. The first is genre prediction using the Bert classifier from a given book description. The other one is using the Facebook pre-trained model (BART) and predicting summarization text from given book descriptions. Using this predicted summarization text(obtained from the BART model) as ground truth and the original data set, we have trained the LSTM model and predicted summary for the given book description. Using this predicted summary from the LSTM model as input to the Bert classifier, the genre is predicted from the summarized book description.

![flow (1)](https://github.com/koushikreddykonda/BookGenrePredictionNLP/assets/122440945/db45b8ce-02dd-4489-b8c9-24fb74719ca7)


#### 4.1.1 Genre Classification

We used a pre-trained BERT model from the Hugging Face Transformers library. The training data is prepared by tokenizing each summary using the BERT tokenizer and then creating input encodings containing token IDs, attention masks, and token type IDs. The matrices generated are one-hot encoded labels. The training data is split into training and validation sets. The model is trained using the training dataset for 10 epochs with a batch size 16 and a validation dataset. The training stops early if the validation loss does not improve after 3 epochs. Genres are predicted by using a test dataset.

#### 4.1.2 Text Summarization

In this project, we used a pre-trained Facebook BART model to generate the summary from the original description. Now we used this generated summary as ground truth values. Now, we designed the LSTM model to predict a summary for the original book description. The LSTM model is trained on the input dataset, and the corresponding summary is generated from the Facebook BART model for training data. Now, we generated a summary for test data by using the Facebook BART model and compared this with the summary predicted by the LSTM model. By using cosine similarity we tried to measure the amount of similarity between two summaries and obtained 0.31. We used pre-trained Google’s PEGASUS model to generate a summary for test data and compared this with the summary predicted by the LSTM model. By using cosine similarity we tried to measure the amount of similarity between two summaries and obtained 0.23. Now, from the predicted summaries obtained from the LSTM model, we tried to predict the genre labels using the BERT classifier. 

#### 4.1.3 Model Building

Encoder: The input to the encoder is a sequence of integers that represent the book descriptions. The input is first passed through an embedding layer that maps the integers to dense vectors of fixed size. The embedding layer is initialized with pre-trained word embeddings. The output of the embedding layer is then passed through a bidirectional LSTM layer, which processes the sequence in both forward and backward directions. 
The bidirectional LSTM layer returns four output tensors:
• enc-output: the sequence of output vectors for each time step in the input sequence 
• enc-fh: the final hidden state of the forward LSTM layer
• enc-fc: the final cell state of the forward LSTM layer
• enc-bh: the final hidden state of the backward LSTM layer
• enc-bc: the final cell state of the backward LSTM layer
• The final hidden and cell states of the LSTM layers are concatenated to produce the enc-h and enc-c tensors, respectively.
Decoder: The input to the decoder is a sequence of integers representing the predicted summaries. The input is first passed through an embedding layer that maps the integers to dense vectors of fixed size. The embedding layer is also initialized with pre-trained word embeddings. The output of the embedding layer is then passed through an LSTM layer that is initialized with the final hidden and cell states of the encoder. 
The LSTM layer returns three output tensors:
• dec-outputs: the sequence of output vectors for each time step in the input sequence
• dec-hidden: the final hidden state of the LSTM layer
• dec-cell: the final cell state of the LSTM layer
• The dec-outputs tensor is passed through a dense layer with a softmax activation function to generate the output sequence of predicted summaries.
Model: The model is defined using the Keras functional API, allowing more flexibility in defining complex models. 
The model takes two inputs: the sequence of book descriptions and the sequence of predicted summaries. The model outputs the sequence of predicted summaries. The
The model function creates the model, with the inputs and outputs specified as lists.

## 5 Experimental Results

### 5.1 Dataset

The dataset we utilized for our project is derived from a November 2, 2012, dump of the English language Wikipedia, containing information on over 16,000 books. The dataset includes several columns, including Wikipedia article ID, book title, author, publication date, book genres, and plot summary. We initially performed data cleaning by removing unnecessary columns, such as Freebase ID, and implemented several techniques to preprocess the data. During our analysis, we identified that the dataset did not have corresponding summarized texts for training. Therefore, we attempted to add a column with the summarized texts to improve our results.
Additionally, we observed that each book had multiple genres associated with it. Initially, we attempted to predict the genre using all the genres. Still, due to the computational complexity and low prediction accuracy, we limited our analysis to books with one, two, three, or four genres. Furthermore, we noticed that books with more than four genres were scarce, and including them in our analysis led to unstable results. By limiting the dataset to books with up to four genres and performing further cleaning and preprocessing, we reduced the dataset to around 4,000 entries, allowing us to achieve improved scores and results. Overall, by utilizing various data-cleaning techniques and limiting our analysis to specific genres and several genres, we were able to optimize the dataset for our specific project and improve our results.

### 5.2 Genre Classification

In our project, one of the primary tasks was to predict the genre of a book based on its description. To achieve this, we explored different approaches
and techniques, such as traditional machine learning models and deep learning models. However, we found that BERT (Bidirectional Encoder Representations from Transformers) provided the best results compared to other models. BERT is a pre-trained transformer-based neural network architecture that can be fine-tuned for specific NLP tasks such as classification, question-answering, and text generation. The transformer architecture of BERT enables it to effectively capture the contextual information of a given text, which is crucial for predicting the genre of a book based on its description. By fine-tuning the pre-trained BERT model on our book dataset, we achieved better results in predicting the genre of books. Overall, we found that BERT’s ability to capture contextual information of a given text was instrumental in providing better genre classification results than other models we explored.

### 5.3 Text Summarization

At the outset of our project, we faced the challenge of selecting an optimal summarization technique. While we initially considered Extractive Summarization, we encountered limitations in inadequate cosine similarity scores and other performance metrics. As a result, we opted for Abstractive Summarization, which deploys an Encoder-Decoder framework featuring a Bidirectional LSTM to encode the input text and generate a practical summary. This approach enabled us to
enhance the performance of our model and attain better results.

### 5.4 Evaluation Protocols

We utilized a standard metric, Cosine Similarity, to evaluate the summarization task. The Cosine Similarity is calculated by measuring the similarity between the two summaries using the cosine similarity metric. For the prediction task, we utilized the standard metric of accuracy. This metric is commonly
used in classification to measure the accuracy and precision of the model’s predictions.

### 5.5 Results and Discussion

The genre prediction results using the original book description and summarized description yielded an accuracy of 58%. The genre prediction results using 
the LSTM summarized description yielded an accuracy of 27%. These results are depicted in the Table 5.5.2. The cosine similarity scores between LSTM-generated summaries and those generated by Facebook BART and Google Pegasus were 0.31 and 0.23, respectively. These results are shown in Table 5.5.1.

Model BART Pegasus <br/>
LSTM 0.31 0.23 <br/>
Table 5.5.1: Cosine Similarity <br/>
<br/>

<br/>
Metric Classifier-1 Classifier-2 <br/>
Accuracy 58% 27% <br/>
Table 5.5.2: Accuracy <br/>
<br/>

Classifier-1: Bert Model on the original book description <br/>
Classifier-2: Bert Model on LSTM predicted summary <br/>

## 6 Conclusions and future work

Based on the results obtained from our project, it can be concluded that summarization techniques have a significant impact on the accuracy of genre prediction. Our LSTM-based abstractive summarization model generated summaries with lower cosine similarity than pre-trained models such as Facebook BART and Google Pegasus. Moreover, the accuracy of genre prediction dropped significantly when using the LSTM-produced summaries compared to the original descriptions. This indicates that the summarization model needs further improvement. In future work, the performance of the summarization model can be improved by fine-tuning the specific domain of book descriptions. Additionally, other pre-trained models for summarization can be explored to achieve better results. Furthermore,
other NLP techniques, such as sentiment analysis and entity recognition, can also be explored to improve genre classification accuracy. Finally,
it’s worth noting that the limited computational resources used in this study may have contributed to the lower accuracy of the LSTM model. With a larger dataset of 16,000 records and more powerful computing resources, we may have achieved even better results.

## 7 References

-  Shikha Gupta, Mohit Agarwal, and Satbir Jain. 2019. Automated genre classification of books using machine learning and natural language processing. In 2019 9th International Conference on Cloud Computing, Data Science Engineering (Confluence), pages 269–272.

-  Eran Ozsarfati, Egemen Sahin, Can Jozef Saul, and Alper Yilmaz. 2019. Book genre classification based on titles with comparative machine learning algorithms. In 2019 IEEE 4th International Conference on Computer and Communication Systems (ICCCS), pages 14–20.

-  Rahul, Surabhi Adhikari, and Monika. 2020. NLP-based machine learning approaches for text summarization. In the 2020 Fourth International Conference on Comput-
ing Methodologies and Communication (ICCMC), pages 535–538.

-  Parilkumar Shiroya. 2021. Book genre categorization using machine learning algorithms (k-nearest neighbor, support vector machine, and logistic regression)
using customized datasets.
