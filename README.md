# About

The repository is created to support blogpost on Custom Sagemakers. There are three steps needed to deploy the solution in AWS.

1. Train&deploy custom model with SageMaker
2. Create API that would use SageMaker model
3. Build and deploy a Flask application which would take the tweets and return results of API


Step 1 is covered in the blogpost. Steps [2](https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/) and [3](https://medium.com/techfront/step-by-step-visual-guide-on-deploying-a-flask-application-on-aws-ec2-8e3e8b82c4f7) available in corresponding tutorials. During step 3 it is possible to use the code of the Flask application which is stored in the folder EC2_Flask_app/.

# Parameters to set up

Please set the following parameters:

1. 'PATH_TO_DATA' in file SageMaker/mysrc/serve.py
2. url in file EC2_Flask_app/app.py
