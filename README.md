terraform-aws-ses-to-gmail
==========================

Configure SES to route traffic to a preauthorized Gmail account with Terraform.

Usage
-----

For personal use:

1. Configure SES for receiving mail, and create the appropriate `MX` records on your domain(s)
2. Create the module somewhere (and configure a Terraform state/handle other boilerplate)
3. Instantiate this module:

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

4. Create a new [GCP project](https://console.cloud.google.com/projectcreate), and
   [configure](https://console.cloud.google.com/apis/credentials) a new OAuth 2.0 Client ID pair.
5. Use the [OAuth Playground](https://developers.google.com/oauthplayground/) to get a refresh token
   for your personal Google account with the `https://www.googleapis.com/auth/gmail.insert` and
   `https://www.googleapis.com/auth/gmail.modify` scopes. You'll can use the gear icon to provide
   your own OAuth credentials, which simplifies this process.
6. Plug your new client ID, client secret, and refresh token into the `google_oauth` object in the
   module (be careful with those secrets!)
5. `terraform apply`

Limitations
-----------

The Lambda currently does not attempt to deduplicate emails it inserts into Gmail, which helps avoid
granting the Lambda read permissions to your email. If we needed to deduplicate incoming emails,
we'd either have to keep a record of emails we've received somewhere (money + complexity), or we'd
need to search via Gmail's API for an existing copy of the email.

The Lambda also does not attempt to pass Gmail's SPF checks; Gmail sees the message as being sent
from some random Amazon, even though it's being uploaded via the Gmail API.

The Lambda stores your credentials in plaintext environment variables.
