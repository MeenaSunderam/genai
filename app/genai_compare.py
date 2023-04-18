import streamlit as st
import boto3
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import requests
from requests.auth import HTTPBasicAuth
import re

from cohere_sagemaker import Client
from sagemaker import KNNPredictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

### account info
ACCESS_KEY = 'AKIA5OK7PVRJHLNHG5FL'
SECRET_KEY = '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'
region = 'us-east-1'

# general constants
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
newline, bold, unbold = "\n", "\033[1m", "\033[0m"

parameters = {
    "max_length": 200,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.95,
    "do_sample": False,
    "temperature": 1,
}

#base model end points
cohere_endpoint = 'cohere-gpt-medium'
flant5_endpoint = 'flan-t5-xxl'

#domain adapted endpoints
stablediffuse_endpoint = 'stable-diffusion-v2-1-base'

#AWS Domain RAG
knn_endpoint = 'jumpstart-example-knn-2023-04-14-14-20-53-714'
endpoint_name_embed = 'jumpstart-example-huggingface-textembed-2023-04-14-14-02-14-427'

#Legal Domain RAG
legal_cohere_model = 'cohere-medium-1681788124'

# Get yaml information
with open('/Users/thandavm/work/strategic_accounts/ai_summit/streamlit/final_demo/config.yml', 'r') as file:
    config = yaml.safe_load(file)

es_username = config['credentials']['username']
es_password = config['credentials']['password']

domain_endpoint = config['domain']['endpoint']
domain_index = config['domain']['index']

URL = f'{domain_endpoint}/{domain_index}/_search'


sagemaker_runtime_client = boto3.client('runtime.sagemaker',
                                aws_access_key_id=ACCESS_KEY,
                                aws_secret_access_key=SECRET_KEY,
                                region_name=region)

sagemaker_client = boto3.client('sagemaker',
                                aws_access_key_id=ACCESS_KEY,
                                aws_secret_access_key=SECRET_KEY,
                                region_name=region)

### Functions
endpointlist = []
def get_endpoints():
    response = sagemaker_client.list_endpoints()
    for endpoint in response['Endpoints']:
        endpointlist.append(endpoint['EndpointName'])

def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type='application/json'):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json)
    return response

def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    generated_text = model_predictions['generated_texts']
    return generated_text

def clean_text(text):
    # Use regular expression to match and remove any trailing characters after the last period.
    cleaned_text = re.sub(r'\.[^\.]*$', '.', text)
    return cleaned_text

def construct_context(context_predictions_arr, df_knowledge) -> str:
    chosen_sections = []
    chosen_sections_len = 0

    for index in context_predictions_arr:
        # Add contexts until we run out of space.
        document_section = df_knowledge.loc[index]
        chosen_sections_len += len(document_section) + 2
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))
    concatenated_doc = "".join(chosen_sections)
    print(
        f"With maximum sequence length {MAX_SECTION_LEN}, selected top {len(chosen_sections)} document sections: {concatenated_doc}"
    )
    return concatenated_doc

def build_embed_table(df_knowledge, endpoint_name_embed, col_name_4_embed, batch_size=10):
    res_embed = []
    N = df_knowledge.shape[0]
    for idx in tqdm(range(0, N, batch_size)):
        content = df_knowledge.loc[idx : (idx + batch_size - 1)][
            col_name_4_embed
        ].tolist()  ## minus -1 as pandas loc slicing is end-inclusive
        payload = {"text_inputs": content}
        query_response = query_endpoint_with_json_payload(
            json.dumps(payload).encode("utf-8"), endpoint_name_embed
        )
        generated_embed = parse_response_multiple_texts(query_response)
        res_embed.extend(generated_embed)
    res_embed_df = pd.DataFrame(res_embed)
    return res_embed_df


df_knowledge = pd.read_csv("/Users/thandavm/work/strategic_accounts/ai_summit/streamlit/final_demo/Amazon_SageMaker_FAQs.csv", header=None, names=["Question", "Answer"])
df_knowledge.drop(["Question"], axis=1, inplace=True)

### Right Column
with st.sidebar:    
    st.subheader("Base Foundation Models")
    cohere = st.checkbox("cohere medium", value= False)
    flant5 = st.checkbox("flan t5 xxl", value= False)
        
    st.subheader("Domain Adapted Models")
    stablediffuse = st.checkbox("stable diffusion 2.1", value= False)
    
    st.subheader("Retrieval assisted gen Models")
    aws_domain = st.checkbox("AWS-Domain", value= False)
    legal_domain = st.checkbox("Legal Domain", value= False)
    
    st.subheader("Model Configurations")
    temperature = st.slider('Temperature', 0.0, 2.0, 0.9, 0.1 )
    tokens = st.slider('Tokens', 50, 400, 100, 10)
    return_seq = st.slider('Return Sequence', 1, 5, 1, 1 )

### Left Column
input_value = st.text_area('Input: ', placeholder='Ask me anything ...', key='prompt', height =  250, max_chars= 5000)
submit_button = st.button("Generate")

col1, col2 = st.columns(2)

if submit_button:
    with col1:
        if cohere:
            co = Client(endpoint_name=cohere_endpoint)
            response = co.generate(prompt=input_value, max_tokens=tokens, temperature=temperature)
            st.text_area(label='cohere-gpt-medium: ', height =  250, value = response.generations[0].text)
            
        if stablediffuse:
            response = sagemaker_runtime_client.invoke_endpoint(EndpointName=stablediffuse_endpoint, 
                                Body=input_value, 
                                ContentType='application/x-text')
            
            response_body = json.loads(response['Body'].read().decode())
            generated_image = response_body['generated_image']
            plt.figure(figsize=(12, 12))
            st.image(np.array(generated_image))
    
    with col2: 
        if flant5:
            parameters = {"max_length":tokens, "num_return_sequences":return_seq, "top_k":50, "top_p":0.95, "do_sample":True}
            input_text = "Answer based on context:" + "\n\n" + input_value
            payload = {"text_inputs": input_text, **parameters}    
            query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'), endpoint_name=flant5_endpoint)
            generated_text = parse_response_multiple_texts(query_response)
            st.text_area(label='flan-t5-xxl: ', height =  250, value = generated_text)
    
        if aws_domain:
            start = time.time()
            #question = "Which instances can I use with Managed Spot Training in SageMaker?"
            question = input_value

            ## embed the input
            query_response = query_endpoint_with_json_payload(
                question, endpoint_name_embed, content_type="application/x-text"
            )            
            embed_model_predict = json.loads(query_response["Body"].read())
            question_embedding = embed_model_predict["embedding"]
            
            ## calling KNN
            knn_predictor = KNNPredictor(knn_endpoint)
            knn_predictor.serializer = CSVSerializer()
            knn_predictor.deserializer = JSONDeserializer()
                
            response = knn_predictor.predict(
                np.array(question_embedding),
                initial_args={"ContentType": "text/csv", "Accept": "application/json; verbose=true"},
            )

            context_predictions_arr = response["predictions"][0]["labels"]
            context_embed_retrieve = construct_context(context_predictions_arr, df_knowledge["Answer"])

            ## Call the foundation model
            prompt =  "Answer based on context:" + "\n\n" + context_embed_retrieve + "\n\n" + question

            payload = {"text_inputs": prompt, **parameters}
            query_response = query_endpoint_with_json_payload(
                json.dumps(payload).encode("utf-8"), endpoint_name=flant5_endpoint)

            model_predictions = json.loads(query_response["Body"].read())
            
            generated_text = model_predictions["generated_texts"]
            st.text_area(label='RAG Model: ', height =  250, value = generated_text[0])
            
        if legal_domain:
            cohere_client = Client(endpoint_name= legal_cohere_model)
            payload = {'text_inputs': [input_value]}
            payload = json.dumps(payload).encode('utf-8')
            response = sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name_embed, 
                                                        ContentType='application/json', 
                                                        Body=payload)
            body = json.loads(response['Body'].read())
            embedding = body['embedding'][0]

            K = 3  # Retrieve Top 3 matching context

            query = {
                'size': K,
                'query': {
                    'knn': {
                        'embedding': {
                            'vector': embedding,
                            'k': K
                        }
                    }
                }
            }

            response = requests.post(URL, auth=HTTPBasicAuth(es_username, es_password), json=query)
            response_json = response.json()
            hits = response_json['hits']['hits']
            
            for hit in hits:
                score = hit['_score']
                passage = hit['_source']['passage']
                doc_id = hit['_source']['doc_id']
                passage_id = hit['_source']['passage_id']
                qa_prompt = f'Context: {passage}\nQuestion: {input_value}\nAnswer:'

                report = []
                res_box = st.empty()
                
                response = cohere_client.generate(prompt=qa_prompt, 
                                                max_tokens=64, 
                                                temperature=0.5, 
                                                return_likelihoods='GENERATION')
                
                answer = response.generations[0].text.strip().replace('\n', '')
                answer = clean_text(answer)
                
                if len(answer) > 0:
                    res_box.markdown(f'**Answer:**\n*{answer}*')

                res_box = st.empty()
                res_box.markdown(f'**Reference**:\n*Document = {doc_id} | Passage = {passage_id} | Score = {score}*')
                res_box = st.empty()
                st.markdown('----')