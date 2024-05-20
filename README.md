WIP....

# Aspect-Based-Sentiment-Analysis

Sentiment analysis is the computational study of opinions, sentiments and subjectivity embedded in user-generated textual document. Usually, the text is classified into positive, negative or neutral sentiments. The most interesting user-generated textual documents are reviews about entities, prod- ucts or services. These use-generated textual documents are of interest to organizations that own or provide the services; supermarkets, movie produces, restaurants, etc. A more fine-grained analysis of the textual document involves identifying various aspects of the main entity in the document and classifying the opinion expressed about the identified aspects. This type of analysis is termed, aspect-based sentiment analysis (ABSA) and it has gained popularity and importance in the last decade.


## Architecture
Exploiting  BERT for Aspect-Based Sentiment Analysis tasks.

todo

ABSA typically involves two sub-tasks:
* Aspect-Based Text Extraction (ABTE): The goal of ABTE is to identify and extract the aspects or attributes mentioned in the text. It is formulated as a sequence labeling task, where each word in the text is assigned a label indicating whether it belongs to a particular aspect or not.
* Aspect-Based Sentiment Analysis (ABSA): The goal of ABSA is to determine the sentiment polarity (positive, negative, or neutral) expressed towards each identified aspect. It is formulated as a classification task, where the sentiment polarity is assigned to each aspect extracted from the text.

## Approach
### Data Pre Label @omolojakazeem@gmail.com
 <img width="1221" alt="image" src="https://github.com/SequinYF/Aspect-Based-Sentiment-Analysis/assets/19517164/97868851-f848-4105-b0cd-1ec118bc4dd9">

### Dataset

https://ai.stanford.edu/~amaas/data/sentiment

### Evaluation Metrics

## Running Models
### Requirements

`pip install -t requirements.txt`

### Dataset

Change the path of the dataset in `consts.py`. Data need to be formatted as in Data Pre Label.

Examples: `todo`

### ABTE

1. `train.sh ABTE`
2. `prediction.sh ABTE`

#### Plot

Related Paper: @https://github.com/aminahjaved @usmanishaq589@gmail.com

contribution：
@

Ref：
* https://github.com/nicolezattarin/BERT-Aspect-Based-Sentiment-Analysis
* https://github.dev/1tangerine1day/Aspect-Term-Extraction-and-Analysis
* https://github.com/lixin4ever/BERT-E2E-ABSA
