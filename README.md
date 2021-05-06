terraform-aws-ses-to-gmail
==========================

Configure SES to route traffic to a preauthorized Gmail account with Terraform.

Usage
-----

For personal use:

```hcl
module "ses-to-gmail" {
 source  = "github.com/skeggse/terraform-aws-ses-to-gmail"

 name = "terraform-email-pipe" # Prefix for all the resources we create.

 recipients  = ["mydomain.com"] # Can be individual addresses or whole domains.
 google_oauth = {
   client_id     = "CLIENT_ID_FROM_SECURE_SOURCE"
   client_secret = "CLIENT_SECRET_FROM_SECURE_SOURCE"
   refresh_token = "REFRESH_TOKEN_FROM_SECURE_SOURCE"
 }

 ses_rule_set_id = "default-rule-set" # Optional.
}
```

Limitations
-----------

The Lambda currently does not attempt to deduplicate emails it inserts into Gmail, which helps avoid
granting the Lambda read permissions to your email. If we needed to deduplicate incoming emails,
we'd either have to keep a record of emails we've received somewhere (money + complexity), or we'd
need to search via Gmail's API for an existing copy of the email.

The Lambda also does not attempt to pass Gmail's SPF checks; Gmail sees the message as being sent
from some random Amazon, even though it's being uploaded via the Gmail API.
