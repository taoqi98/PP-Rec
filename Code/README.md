# PP-Rec
- Code of our PP-Rec model

# Data Preparation
- Codes in this project can be used on MIND dataset (https://msnews.github.io/index.html) for expseriments
- All recommendation data are stored in data-root-path
- We used the glove.840B.300d embedding vecrors in https://nlp.stanford.edu/projects/glove/
- The embedding file should be stored in embedding\_path\glove.840B.300d.txt
- The meta data of entity (including news entities, and pre-trained transE embeddings) should be stored in KG\_root\_path

# Code Files
- utils.py: containing some util functions
- preprocess.py: containing functions to preprocess data
- NewsContent.py: Class for loading and managing news data
- UserContent.py: Class for loading and managing user data
- PEGenerator.py: data generator for model training and evaluation
- models.py: containing basic models such as Attention network
- Encoders.py: containing encoders or predictors in our model
- Main.ipynb: containing codes for model training and evaluation