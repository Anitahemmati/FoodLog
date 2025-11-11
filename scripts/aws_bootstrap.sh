#!/usr/bin/env bash
#
# Provision the AWS resources required by the Food Log backend.
# The script is idempotent: it will skip resources that already exist.
# Required environment variables:
#   - AWS_BUCKET_NAME (S3 bucket for uploads)
# Optional overrides:
#   - AWS_REGION               (defaults to us-east-1)
#   - AWS_PROFILE              (uses default CLI profile if unset)
#   - AWS_DYNAMODB_TABLE       (defaults to FoodLogHistory)
#   - AWS_USERS_TABLE          (defaults to FoodLogUsers)
#   - ECR_REPOSITORY           (defaults to foodlog-backend)
#   - ENABLE_PUBLIC_READ       (true|false, defaults to true)
set -euo pipefail

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install it from https://aws.amazon.com/cli/ and retry." >&2
  exit 1
fi

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-}"
AWS_BUCKET_NAME="${AWS_BUCKET_NAME:-}"
AWS_DYNAMODB_TABLE="${AWS_DYNAMODB_TABLE:-FoodLogHistory}"
AWS_USERS_TABLE="${AWS_USERS_TABLE:-FoodLogUsers}"
ECR_REPOSITORY="${ECR_REPOSITORY:-foodlog-backend}"
ENABLE_PUBLIC_READ="${ENABLE_PUBLIC_READ:-true}"

if [[ -z "${AWS_BUCKET_NAME}" ]]; then
  echo "AWS_BUCKET_NAME must be set before running this script." >&2
  exit 1
fi

# macOS ships with Bash 3.2 by default, so avoid `${var,,}` expansion.
ENABLE_PUBLIC_READ_NORMALIZED=$(printf '%s' "${ENABLE_PUBLIC_READ}" | tr '[:upper:]' '[:lower:]')

AWS_ARGS=( "--region" "${AWS_REGION}" )
if [[ -n "${AWS_PROFILE}" ]]; then
  AWS_ARGS+=( "--profile" "${AWS_PROFILE}" )
fi

aws_cmd() {
  aws "${AWS_ARGS[@]}" "$@"
}

echo "Using region: ${AWS_REGION}"
[[ -n "${AWS_PROFILE}" ]] && echo "Using profile: ${AWS_PROFILE}"

# --- S3 bucket ---
echo "Checking S3 bucket: ${AWS_BUCKET_NAME}"
if aws_cmd s3api head-bucket --bucket "${AWS_BUCKET_NAME}" >/dev/null 2>&1; then
  echo "Bucket already exists."
else
  echo "Creating bucket..."
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws_cmd s3api create-bucket --bucket "${AWS_BUCKET_NAME}"
  else
    aws_cmd s3api create-bucket \
      --bucket "${AWS_BUCKET_NAME}" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  fi
fi

if [[ "${ENABLE_PUBLIC_READ_NORMALIZED}" == "true" ]]; then
  echo "Ensuring bucket allows public read (required for direct image URLs)."
  aws_cmd s3api put-public-access-block \
    --bucket "${AWS_BUCKET_NAME}" \
    --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false

  BUCKET_POLICY=$(cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::__BUCKET__/*"
    }
  ]
}
JSON
)
  BUCKET_POLICY="${BUCKET_POLICY/__BUCKET__/${AWS_BUCKET_NAME}}"
  aws_cmd s3api put-bucket-policy --bucket "${AWS_BUCKET_NAME}" --policy "${BUCKET_POLICY}"

  CORS_CONFIG=$(cat <<'JSON'
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST"],
      "AllowedOrigins": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}
JSON
)
  aws_cmd s3api put-bucket-cors --bucket "${AWS_BUCKET_NAME}" --cors-configuration "${CORS_CONFIG}"
else
  echo "Skipping public read configuration for bucket."
fi

# --- DynamoDB tables ---
create_dynamo_table() {
  local table_name="$1"
  local key_name="$2"
  echo "Checking DynamoDB table: ${table_name}"
  if aws_cmd dynamodb describe-table --table-name "${table_name}" >/dev/null 2>&1; then
    echo "Table already exists."
    return
  fi

  echo "Creating DynamoDB table: ${table_name}"
  aws_cmd dynamodb create-table \
    --table-name "${table_name}" \
    --attribute-definitions "AttributeName=${key_name},AttributeType=S" \
    --key-schema "AttributeName=${key_name},KeyType=HASH" \
    --billing-mode PAY_PER_REQUEST

  echo "Waiting for table to become active..."
  aws_cmd dynamodb wait table-exists --table-name "${table_name}"
}

create_dynamo_table "${AWS_DYNAMODB_TABLE}" "id"
create_dynamo_table "${AWS_USERS_TABLE}" "email"

# --- ECR repository ---
if [[ -n "${ECR_REPOSITORY}" ]]; then
  echo "Checking ECR repository: ${ECR_REPOSITORY}"
  if aws_cmd ecr describe-repositories --repository-names "${ECR_REPOSITORY}" >/dev/null 2>&1; then
    echo "ECR repository already exists."
  else
    echo "Creating ECR repository..."
    aws_cmd ecr create-repository --repository-name "${ECR_REPOSITORY}" --image-scanning-configuration scanOnPush=true >/dev/null
  fi
fi

echo "Bootstrap complete."
