# DSS_CTBA
A dual-branch neural network based on DNA sequence and structure for predicting transcription factor binding sites

DSS_ CTBA is a dual branch network model used to predict transcription factor binding sites, which is mainly divided  
into two parts: first, convolution and attention mechanisms are applied to DNA sequence information to extract 
high-order sequence features; Secondly, the Transformer module and BiLSTM module were added to extract 
higher-order structural features for DNA structural information. Then input them separately into the output module, 
and finally input the output results into two classifiers for averaging.

# Requirements 
 * Python 3.8 or higher 
 * PyTorch 1.8.0 or higher 
 * numpy
 * sklearn

# Data 
165 ChIP seq datasets were collected from 690 datasets in the Encyclopedia of DNA Elements (ENCODE) database as 
the baseline dataset, which includes 29 transcription factors from different cell lines. The length of DNA sequences in 
each dataset is 101 base pairs. The datasets "wgEncodeAwgTfbsHaibHepg2Fosl2V0416101UniPk" and
"wgEncodeAwgTfbsHaibK562Fosl1sc183V0416101UniPk" in the code are examples, where each dataset contains
sequence and structural information, which is further divided into training data and test data.

# Running the Code 
 * Build a good Python environment
 * From "Train. py" replace the value of the data_name attribute with the name of the dataset you want to run
 * Run "Train. py" to train data and obtain results, which will be directly written to the result.txt file 
