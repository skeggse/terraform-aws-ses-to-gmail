variable "name" {
  type = string
}

variable "deploy_bucket" {
  type = string
}

variable "s3_bucket_name" {
  type    = string
  default = null
}

variable "s3_bucket_prefix" {
  type    = string
  default = null
}

variable "ses_rule_set_name" {
  type    = string
  default = "default-rule-set"
}

variable "extra_gmail_label_ids" {
  type    = set(string)
  default = []

  validation {
    condition = (
      length([
        for label_id in var.extra_gmail_label_ids :
        label_id if !can(regex(":", label_id))
      ]) == length(var.extra_gmail_label_ids)
    )
    error_message = "Label IDs cannot contain the ':' character, as it is used as a delimiter."
  }
}

variable "recipients" {
  type = set(string)
}

variable "google_oauth" {
  type = object({
    client_id        = string
    secret_parameter = string
    token_parameter  = string
  })
}

variable "inbound_tls_policy" {
  type    = string
  default = "Optional"

  validation {
    condition     = contains(["Optional", "Required"], var.inbound_tls_policy)
    error_message = "The TLS policy must be one of 'Optional', 'Required'."
  }
}

# TODO: support dead letter queue for lambda failures
# variable "configure_dead_letter_queue" {
#   type = bool
# }
