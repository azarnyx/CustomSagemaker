import os
import json
import numpy as np
from joblib import load
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump, load
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, TFAutoModel

# Specify that endpoint accept JSON
JSON_CONTENT_TYPE = 'application/json'

# parameters for HuggingFace library
MODEL = "cardiffnlp/twitter-roberta-base"
TOKENIZER_EMB = AutoTokenizer.from_pretrained(MODEL)
MODEL_EMB = AutoModel.from_pretrained(MODEL)


# functions for HuggingFace library
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_embedding(text):
    text = preprocess(text)
    encoded_input = TOKENIZER_EMB(text, return_tensors='pt')
    features = MODEL_EMB(**encoded_input)
    features = features[0].detach().cpu().numpy() 
    features_mean = np.mean(features[0], axis=0) 
    return features_mean


####### DEPLOYMENT FUNCTIONS #############################

def model_fn(model_dir):
    clf = load(os.path.join(model_dir, 'sklearnclf.joblib'))
    return clf


def predict_fn(input, model):
    proba = model.predict_proba(input)
    return json.dumps({
        "proba": str(list(proba[0]))
    })


def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    # process an jsonlines uploaded to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        request_body = json.loads(request_body)
        st = request_body["text"]
        return get_embedding(st).reshape(1,-1)
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))


def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))
##############################################################


####### TRAINING PART ########################################
if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    
    # Derive embedings
    initial_df = pd.read_csv("s3://dmitriaz/data/train_data_cleaning.csv", index_col=[0])
    embed_df = initial_df.text.apply(get_embedding)
    embed_df = pd.DataFrame(embed_df.to_list(), index= embed_df.index)

    # Train model
    X = embed_df
    y = initial_df.target
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
    
    
    classif = RandomForestClassifier(n_estimators=80, max_depth=8)
    
    
    classif.fit(X_train, y_train)
    
    p_train = classif.predict(X_train)
    p_test = classif.predict(X_test)
    
    dump(classif, os.path.join(args.model_dir, 'sklearnclf.joblib'))
##############################################################
