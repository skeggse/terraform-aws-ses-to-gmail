locals {
  local_arn_infix = "${data.aws_region.current.name}:${local.account_id}"
}

data "aws_region" "current" {}

data "aws_iam_policy_document" "function-policy" {
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.function-logs.arn}:*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectTagging",
      "s3:ListBucket",
      "s3:PutObjectTagging",
    ]
    resources = [
      "arn:aws:s3:::${aws_s3_bucket.storage.bucket}",
      "arn:aws:s3:::${aws_s3_bucket.storage.bucket}/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameter",
    ]
    resources = [
      for param in [
        var.google_oauth.secret_parameter,
        var.google_oauth.token_parameter,
      ] :
      "arn:aws:ssm:${local.local_arn_infix}:parameter/${trimprefix(param, "/")}"
    ]
  }
}

data "aws_iam_policy_document" "assume-function-policy" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "function-policy" {
  name   = "${var.name}-policy"
  policy = data.aws_iam_policy_document.function-policy.json
}

resource "aws_iam_role" "function-role" {
  name               = var.name
  assume_role_policy = data.aws_iam_policy_document.assume-function-policy.json
  description        = "Allow the ${local.function_name} Lambda to read messages from S3 and mark them for later deletion."
}

resource "aws_iam_role_policy_attachment" "function" {
  role       = aws_iam_role.function-role.name
  policy_arn = aws_iam_policy.function-policy.arn
}
