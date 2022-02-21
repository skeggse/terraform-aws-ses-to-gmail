terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4"
    }
  }
}

locals {
  account_id    = data.aws_caller_identity.current.account_id
  bucket_name   = var.s3_bucket_name == null ? "${var.name}-storage" : var.s3_bucket_name
  function_name = var.name

  bundle_path = "${path.module}/bundle.zip"
}

data "aws_caller_identity" "current" {}

# Allow SES in the current account to store objects in the S3 bucket.
data "aws_iam_policy_document" "storage-policy" {
  statement {
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["arn:aws:s3:::${local.bucket_name}/*"]
    principals {
      type        = "Service"
      identifiers = ["ses.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:Referer"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

# Temporarily holds the (up to 30MB) raw message data.
resource "aws_s3_bucket" "storage" {
  bucket = local.bucket_name
  acl    = "private"
  policy = data.aws_iam_policy_document.storage-policy.json

  lifecycle_rule {
    enabled = true
    id      = "delete-old-forwarded-emails"
    tags = {
      Forwarded = "true"
    }

    expiration {
      days = 7
    }
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "storage-bpa" {
  bucket = aws_s3_bucket.storage.bucket

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudwatch_log_group" "function-logs" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 90
}

data "archive_file" "bundle" {
  type        = "zip"
  output_path = local.bundle_path

  source {
    content  = file("${path.module}/main.py")
    filename = "main.py"
  }
}

data "aws_ssm_parameter" "secret" {
  name            = var.google_oauth.secret_parameter
  with_decryption = false
}

data "aws_ssm_parameter" "token" {
  name            = var.google_oauth.token_parameter
  with_decryption = false
}

module "function" {
  source = "../terraform-modules/lambda"
  # source = "github.com/skeggse/terraform-modules//lambda?ref=main"

  name = local.function_name
  role_arn = module.function_role.arn

  deploy_bucket = var.deploy_bucket
  # TODO: source these from the deploy object.
  handler = "main.lambda_handler"
  runtime = "python3.9"

  timeout = 60
  memory_size = 256

  env_vars = {
    AWS_ACCOUNT_ID = local.account_id

    GOOGLE_CLIENT_ID        = var.google_oauth.client_id
    GOOGLE_SECRET_PARAMETER = data.aws_ssm_parameter.secret.name
    GOOGLE_TOKEN_PARAMETER  = data.aws_ssm_parameter.token.name

    S3_BUCKET = aws_s3_bucket.storage.bucket
    S3_PREFIX = var.s3_bucket_prefix == null ? null : trimsuffix(var.s3_bucket_prefix, "/")

    EXTRA_GMAIL_LABEL_IDS = join(":", var.extra_gmail_label_ids)

    PYTHONPATH = "site-packages"
  }
}

resource "aws_lambda_permission" "ses-invoke" {
  statement_id   = "allowSesInvoke"
  function_name  = module.function.function_arn
  principal      = "ses.amazonaws.com"
  action         = "lambda:InvokeFunction"
  source_account = local.account_id
}

resource "aws_ses_receipt_rule" "store-and-forward" {
  name          = "${var.name}-store-and-forward"
  rule_set_name = var.ses_rule_set_name
  recipients    = var.recipients
  enabled       = true
  scan_enabled  = true

  # This allows SES to _receive_ emails over insecure connections. Your threat model may forbid
  # this.
  tls_policy = var.inbound_tls_policy

  s3_action {
    bucket_name = aws_s3_bucket.storage.bucket
    position    = 1
  }

  lambda_action {
    function_arn    = module.function.invoke_arn
    invocation_type = "Event"
    position        = 2
  }

  lifecycle {
    # Just in case you change the name on this.
    create_before_destroy = true
  }
}
