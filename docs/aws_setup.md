# AWS Deployment Guide

Follow these steps to provision the minimum AWS resources required by the backend and to supply the configuration values the application expects.

## 1. Prerequisites
- AWS account with permissions to manage S3, DynamoDB, and IAM.
- AWS CLI installed locally (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
- Configure credentials with `aws configure` or environment variables.

## 2. Create an S3 bucket for image uploads
```bash
aws s3api create-bucket \
  --bucket food-log-uploads-example \
  --region us-east-1 \
  --create-bucket-configuration LocationConstraint=us-east-1
```

Enable public access strictly for the uploaded objects or configure CloudFront/Presigned URLs if you prefer private buckets. The backend assumes `public-read` uploads.

## 3. Provision DynamoDB tables

### Meal history table
```bash
aws dynamodb create-table \
  --table-name FoodLogHistory \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### Users table
```bash
aws dynamodb create-table \
  --table-name FoodLogUsers \
  --attribute-definitions AttributeName=email,AttributeType=S \
  --key-schema AttributeName=email,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## 4. Configure IAM credentials
Create an IAM user or role with permissions:
- `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject` on the upload bucket.
- `dynamodb:PutItem`, `dynamodb:GetItem`, `dynamodb:Scan` on both tables.

Attach an access key/secret if you are running from EC2 or a local machine. For ECS/Lambda prefer an IAM role.

## 5. Update configuration
Copy `config/aws.example.json` to `config/aws.json` (or add variables to `.env`) and edit:
- `AWS_REGION`: region in which you created the resources.
- `AWS_BUCKET_NAME`: bucket name from step 2.
- `AWS_DYNAMODB_TABLE`: `FoodLogHistory`.
- `AWS_USERS_TABLE`: `FoodLogUsers`.
- `AWS_S3_ACL`: optional object ACL to apply (`public-read` by default, leave empty if the bucket uses Object Ownership "Bucket owner enforced"`).
- `ML_SERVICE_URL`: endpoint for your inference service (set to `stub` for temporary random responses).
- `ML_SERVICE_API_KEY`: optional bearer token for the ML service.
- `JWT_SECRET_KEY`: long random string (`openssl rand -hex 32`).
- `STORAGE_BACKEND`: must be `aws` for production deployments.

## 6. Verify connectivity
Run the app with the AWS configuration:
```bash
export STORAGE_BACKEND=aws
export AWS_REGION=us-east-1
export AWS_BUCKET_NAME=food-log-uploads-example
export AWS_DYNAMODB_TABLE=FoodLogHistory
export AWS_USERS_TABLE=FoodLogUsers
export AWS_S3_ACL=public-read      # Set to empty if ACLs are disabled
export ML_SERVICE_URL=stub  # Replace with real endpoint when ready
export ML_SERVICE_API_KEY=replaceme
export JWT_SECRET_KEY=$(openssl rand -hex 32)
gunicorn -b 0.0.0.0:5000 app:create_app()
```

Hit `/health` and `/history` to ensure the app can talk to AWS. The first call to `/predict` will verify S3 uploads and DynamoDB writes.

## 7. Optional: Infrastructure as code
If you prefer declarative provisioning, translate the steps above into CloudFormation, Terraform, or CDK. The table and bucket configuration above map directly to those templates.
