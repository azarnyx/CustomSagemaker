import os
import io
import boto3
import json

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=json.dumps({"text": payload}),
        ContentType='application/json',
        Accept="application/json"
    )
    print(response)
    result = response['Body'].read().decode()
    print(result)
    result = json.loads(json.loads(result))['proba']
    result = [float(i[:5]) for i in result[1:-1].split(', ')]
    print(result)
    
    return result
