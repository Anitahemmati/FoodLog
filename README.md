# FoodLog Backend (AWS-first)

Production-focused Flask service that powers the FoodLog meal-tracking app. The service is architected to run on AWS (S3 for image storage, DynamoDB for persistence, IAM-backed access). A lightweight local mode (using SQLite + filesystem) is still available when you explicitly switch to it, but every deploy we describe here mirrors the AWS topology we actually use in the cloud.

## Features
- **Multi-tier storage** – user and meal data live in DynamoDB; uploaded photos land in S3 with public-read ACL (or any ACL you configure). Local mode writes to SQLite/filesystem only when you opt into it.
- **Secure auth** – JWT signup/login endpoints with hashed passwords. Tokens expire after `JWT_EXPIRATION_MINUTES` (default 60).
- **Inference pipeline** – `/predict` accepts `multipart/form-data`, enriches EXIF metadata, calls a Hugging Face model (router-based inference API) for classification, and immediately augments the prediction with USDA FoodData Central nutrition (per-ingredient weighting where available). Everything is committed to DynamoDB/S3 in one transaction.
- **Configurable ML fallbacks** – optional external ML microservice (`ML_SERVICE_URL`) plus the lightweight synthetic predictor to guarantee a response when upstream systems are down.
- **Deployment ready** – Dockerized, Render blueprint provided, and AWS credentials consumed directly via environment variables so it can run on ECS, EC2, Lambda + API Gateway, or any container host.

## Architecture Overview
1. Client uploads a meal photo to `/predict`.
2. App stores the file in S3 (`AWS_BUCKET_NAME`) and extracts metadata.
3. Hugging Face classifier labels the meal; predictions under `HF_CONFIDENCE_THRESHOLD` trigger a rejection with `"Sorry! Unfortunately, we cannot analyze your picture!"`.
4. USDA FDC is queried either per ingredient (weighted) or via a single-food lookup.
5. Final payload (image URL, macros, metadata, user) is inserted into DynamoDB (`AWS_DYNAMODB_TABLE`). A DynamoDB `FoodLogUsers` table keeps credentials.

## Prerequisites
- Python 3.11
- AWS account with:
  - S3 bucket (e.g., `foodlog-dev-uploads`)
  - DynamoDB tables (`FoodLogHistory`, `FoodLogUsers`)
  - IAM user or role with `s3:PutObject`, `s3:GetObject`, `dynamodb:GetItem`, `dynamodb:PutItem`, `dynamodb:Scan`
- Hugging Face API token (Inference API access)
- USDA FoodData Central API key

## Environment Variables (AWS deployment)

| Variable | Description |
| --- | --- |
| `STORAGE_BACKEND` | Set to `aws` in production. (`sqlite` only for local prototyping.) |
| `AWS_REGION` | Region for S3 + DynamoDB (`us-east-1` in examples). |
| `AWS_BUCKET_NAME` | S3 bucket for uploads. |
| `AWS_DYNAMODB_TABLE`, `AWS_USERS_TABLE` | DynamoDB table names. |
| `AWS_S3_ACL` | (Optional) ACL applied to uploads (default `public-read`). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | IAM credentials. Example placeholders we use in docs: `AKIAFAKEKEY123456` / `deadbeefdeadbeeffakedummytoken`. Replace with real keys in prod. |
| `UPLOADS_DIR`, `SQLITE_DB_PATH` | Ignored in AWS mode, but required for local fallback. |
| `HF_SPACE_URL` | Defaults to `https://router.huggingface.co/hf-inference/models/nateraw/food`. Legacy `api-inference` URLs are auto-normalised to the router host. |
| `HF_API_TOKEN` | Hugging Face bearer token. |
| `HF_CONFIDENCE_THRESHOLD`, `HF_REJECTION_STATUS_CODE`, `HF_SPACE_TIMEOUT` | Hugging Face tuning knobs. |
| `FDC_API_KEY`, `FDC_PAGE_SIZE`, `FDC_DATA_TYPES`, `FDC_TIMEOUT`, `FDC_BRAND_OWNER` | USDA configuration. |
| `ML_SERVICE_URL`, `ML_SERVICE_API_KEY` | Optional external ML microservice settings. |
| `JWT_SECRET_KEY`, `JWT_EXPIRATION_MINUTES` | JWT signing + expiry. |
| `STORAGE_BACKEND` | Repeated intentionally – keep it `aws` unless you explicitly want SQLite. |

## Deploying on AWS (example)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export STORAGE_BACKEND=aws
export AWS_REGION=us-east-1
export AWS_BUCKET_NAME=foodlog-dev-uploads
export AWS_DYNAMODB_TABLE=FoodLogHistory
export AWS_USERS_TABLE=FoodLogUsers
export AWS_ACCESS_KEY_ID=AKIAFAKEKEY123456          # placeholder for docs
export AWS_SECRET_ACCESS_KEY=deadbeefdeadbeeffakedummytoken  # placeholder
export HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
export FDC_API_KEY=fdc_xxxxxxxxxxxxxxxxx
export JWT_SECRET_KEY=$(openssl rand -hex 32)

gunicorn -b 0.0.0.0:5000 app:create_app()
```

Swap the fake IAM values with your real ones before hitting production. We typically run this containerized (ECS/Fargate or EC2), but the command above works on any host with Python 3.11.

## Local Development

If you just want to hack on the API without AWS credentials:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export STORAGE_BACKEND=sqlite
export JWT_SECRET_KEY=$(openssl rand -hex 32)
export HF_API_TOKEN=hf_dev_token
export FDC_API_KEY=fdc_dev_token
python app.py
```

Uploads go to `./uploads`, data to `./food_history.db`. No S3/DynamoDB interaction occurs in this mode, even though the codebase remains AWS-ready.

## Docker / Render

```bash
docker build -t foodlog-backend .
docker run -p 5000:5000 \
  -e STORAGE_BACKEND=aws \
  -e AWS_REGION=us-east-1 \
  -e AWS_BUCKET_NAME=foodlog-dev-uploads \
  -e AWS_DYNAMODB_TABLE=FoodLogHistory \
  -e AWS_USERS_TABLE=FoodLogUsers \
  -e AWS_ACCESS_KEY_ID=AKIAFAKEKEY123456 \
  -e AWS_SECRET_ACCESS_KEY=deadbeefdeadbeeffakedummytoken \
  -e HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxx \
  -e FDC_API_KEY=fdc_xxxxxxxxxxxxxxxxx \
  foodlog-backend
```

Render deployments can lean on `render.yaml`, which already declares the AWS-oriented env vars (with secret syncing). Attach a 1 GB persistent disk at `/data` if you plan to run SQLite instead.

## API Surface

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/auth/signup` | Create user (stored in DynamoDB or SQLite). |
| `POST` | `/auth/login` | Obtain JWT. |
| `GET` | `/auth/me` | Validate token, return profile. |
| `POST` | `/predict` | Upload image (`image` or `photo` field). Returns label, calories, macros, metadata (HF + USDA), S3 URL, etc. Rejects low-confidence photos with `Sorry! Unfortunately, we cannot analyze your picture!`. |
| `GET` | `/history` | Authenticated; fetches per-user meal history (DynamoDB scan + filter). |
| `GET` | `/meals` | Legacy SQLite listing (kept for backwards compatibility). |
| `POST` | `/save_meal` | Manual meal entry; stored in DB backend. |
| `POST` | `/upload` | Simple upload helper (writes to S3 or local). |
| `GET` | `/health` | Health check with timestamp. |

Full contract lives in `docs/openapi.yaml`.

## Testing

We rely on runtime smoke tests and `python -m compileall app.py` for quick syntax verification. Add your own pytest/flake8 suites as needed.

## Repo Layout

```
app.py                # Flask app + route definitions
api/                  # utilities (Hugging Face client, ML service stub, metadata extraction)
models/, database/    # SQLAlchemy models + helpers
docs/                 # Render guide, AWS setup guide, OpenAPI spec
config/               # sample JSON config
Dockerfile, render.yaml
```

## Notes
- Hugging Face router endpoint is the default; if you mistakenly set `HF_SPACE_URL=https://api-inference.huggingface.co/...`, the code rewrites it to the router host automatically.
- USDA nutrition aggregation prioritizes ingredient-level lookups when weights are known (`HF_INGREDIENT_WEIGHT_OVERRIDES`). Otherwise, the single-food lookup is used, and only if both fail do we fall back to the synthetic predictor.
- Error messaging is user-facing: low-confidence predictions (below `HF_CONFIDENCE_THRESHOLD`) return `Sorry! Unfortunately, we cannot analyze your picture!` and nothing is stored.
