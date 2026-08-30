#!/usr/bin/env bash
set -euo pipefail

# Configuration (override with environment variables as needed)
REGION="${REGION:-us-east-2}"
DB_SECRET_NAME="${DB_SECRET_NAME:-rds/appdb/appuser}"
PUBLIC_DB_SECRET_NAME="${PUBLIC_DB_SECRET_NAME:-rds/appdb/appuser-readonly}"
R2_SECRET_NAME="${R2_SECRET_NAME:-r2/rldb/credentials}"
PUBLIC_R2_SECRET_NAME="${PUBLIC_R2_SECRET_NAME:-r2/rldb/public/credentials}"
ENV_FILE="${ENV_FILE:-./src/egomimic/utils/aws/.egoverse_env}"
BUCKET="${BUCKET:-rldb}"

echo "=== Downloading EgoVerse env from Secrets Manager ==="

SECRET_ARN=""
EFFECTIVE_DB_SECRET_NAME=""
if SECRET_ARN="$(
  aws secretsmanager describe-secret \
    --secret-id "$DB_SECRET_NAME" \
    --region "$REGION" \
    --query 'ARN' \
    --output text 2>/dev/null
)"; then
  EFFECTIVE_DB_SECRET_NAME="$DB_SECRET_NAME"
elif [[ "$PUBLIC_DB_SECRET_NAME" != "$DB_SECRET_NAME" ]] && SECRET_ARN="$(
  aws secretsmanager describe-secret \
    --secret-id "$PUBLIC_DB_SECRET_NAME" \
    --region "$REGION" \
    --query 'ARN' \
    --output text 2>/dev/null
)"; then
  EFFECTIVE_DB_SECRET_NAME="$PUBLIC_DB_SECRET_NAME"
else
  :
fi

if R2_SECRET_JSON="$(
  aws secretsmanager get-secret-value \
    --secret-id "$R2_SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text 2>/dev/null
)"; then
  :
elif [[ "$PUBLIC_R2_SECRET_NAME" != "$R2_SECRET_NAME" ]] && R2_SECRET_JSON="$(
  aws secretsmanager get-secret-value \
    --secret-id "$PUBLIC_R2_SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text 2>/dev/null
)"; then
  R2_SECRET_NAME="$PUBLIC_R2_SECRET_NAME"
else
  echo "Failed to read R2 secret from either $R2_SECRET_NAME or $PUBLIC_R2_SECRET_NAME" >&2
  exit 1
fi

CREDENTIAL_MODE="admin"
if [[ "$R2_SECRET_NAME" == "$PUBLIC_R2_SECRET_NAME" ]] || [[ "$EFFECTIVE_DB_SECRET_NAME" == "$PUBLIC_DB_SECRET_NAME" ]]; then
  CREDENTIAL_MODE="public"
fi

if [[ "$CREDENTIAL_MODE" == "public" ]]; then
  echo "Downloading Public EgoVerse read only credentials"
  echo "Region: $REGION"
  if [[ -n "$EFFECTIVE_DB_SECRET_NAME" ]]; then
    echo "DB secret: $EFFECTIVE_DB_SECRET_NAME"
  fi
  echo "R2 secret: $R2_SECRET_NAME"
else
  echo "Downloading EgoVerse Admin Credentials"
  echo "Region: $REGION"
  if [[ -n "$EFFECTIVE_DB_SECRET_NAME" ]]; then
    echo "DB secret: $EFFECTIVE_DB_SECRET_NAME"
  fi
  echo "R2 secret: $R2_SECRET_NAME"
fi

read -r R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_SESSION_TOKEN AWS_ENDPOINT_URL_S3 < <(
  R2_SECRET_JSON="$R2_SECRET_JSON" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["R2_SECRET_JSON"])
access = payload.get("access_key_id", "")
secret = payload.get("secret_access_key", "")
session = payload.get("session_token", "")
endpoint = payload.get("endpoint_url", "")

if not access or not secret or not endpoint:
    print("Missing required keys in R2 secret JSON.", file=sys.stderr)
    sys.exit(1)

print(access, secret, session or "__EMPTY__", endpoint)
PY
)

{
  if [[ -n "$SECRET_ARN" ]]; then
    printf "SECRETS_ARN=%q\n" "$SECRET_ARN"
  fi
  printf "R2_ACCESS_KEY_ID=%q\n" "$R2_ACCESS_KEY_ID"
  printf "R2_SECRET_ACCESS_KEY=%q\n" "$R2_SECRET_ACCESS_KEY"
  if [[ "$R2_SESSION_TOKEN" != "__EMPTY__" ]]; then
    printf "R2_SESSION_TOKEN=%q\n" "$R2_SESSION_TOKEN"
  fi
  printf "AWS_ENDPOINT_URL_S3=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "R2_ENDPOINT_URL=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "S3_ENDPOINT_URL=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "AWS_DEFAULT_REGION=%q\n" "$REGION"
  printf "BUCKET=%q\n" "$BUCKET"
} >"$ENV_FILE"

chmod 600 "$ENV_FILE"
echo "✅ Wrote runtime environment to $ENV_FILE"
