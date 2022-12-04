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
    resources = ["${module.function.logs_arn}:*"]
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

  dynamic "statement" {
    for_each = range(var.events_sns_topic_arn == null ? 0 : 1)

    content {
      effect = "Allow"
      actions = [
        "sns:Publish",
      ]
      resources = [
        var.events_sns_topic_arn,
      ]
    }
  }
}

module "function_role" {
  source = "github.com/skeggse/terraform-modules//role?ref=main"

  name        = var.name
  description = "the ${local.function_name} Lambda to read messages from S3 and mark them for later deletion."
  policy      = data.aws_iam_policy_document.function-policy.json
  assume_role_principals = [
    {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    },
  ]
}
