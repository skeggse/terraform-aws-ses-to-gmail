terraform-aws-ses-to-gmail
==========================

Configure SES to route traffic to a preauthorized Gmail account with Terraform.

![Technical overview](https://github.com/skeggse/terraform-aws-ses-to-gmail/blob/default/docs/diagram.png)

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
       client_id = "EXAMPLE.apps.googleusercontent.com"
       # Example parameters - you determine where these are stored.
       secret_parameter  = "/Dev/ServiceProviders/GoogleClient"
       refresh_parameter = "/Dev/TenantCredentials/Google"
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
6. Create two new parameters in SSM parameter store: one for the client secret and one for the
   refresh token.
  * For the client secret, provision the json-encoded fields `client_id` and `client_secret`. In the
    example above, you'd create the `/Dev/ServiceProviders/GoogleClient` parameter with a value that
    looks like
    `{"client_id":"EXAMPLE.apps.googleusercontent.com","client_secret":"0cPppYgzfKdHyysI1sPpZF4N"}`.
  * For the refresh token, provision the json-encoded field `refresh_token`. In the example above,
    you'd create the `/Dev/TenantCredentials/Google` parameter with a value that looks like
    `{"refresh_token":"Yy6VnmnpMWdj4zgyLqJ1PQ"}`.
7. Plug the new client ID into the `google_oauth` object in the module, along with the names of the
   SSM parameters provisioned.
8. `terraform apply`

Replace the `refresh_token`
---------------------------

Instead of using the OAuth Playground to get a refresh token once, which may in rare cases cease
functioning, consider using `terraform-aws-oauth2-authenticator` to simplify reauthorization
following the refresh token being revoked:

```hcl
locals {
  client_id        = "EXAMPLE.apps.googleusercontent.com"
  secret_parameter = "/Dev/ServiceProviders/GoogleClient"
}

module "ses-to-gmail" {
  ...

  google_oauth = {
    client_id        = local.client_id
    secret_parameter = local.secret_parameter
    token_parameter  = module.authorizer.service_token_parameters.google
  }
}

module "authorizer" {
  source = "github.com/skeggse/terraform-aws-oauth2-authenticator"

  # Prefix for resource names.
  name             = "authorizer"
  parameter_prefix = "/Dev/TenantCredentials"

  services = {
    google = {
      client_id        = local.client_id
      secret_parameter = local.secret_parameter
      extra_params = {
        access_type = "offline"
      }

      scopes = [
        "https://www.googleapis.com/auth/gmail.insert",
        "https://www.googleapis.com/auth/gmail.modify",
      ]

      authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
      token_endpoint         = "https://oauth2.googleapis.com/token"

      token_endpoint_auth_method = "parameter"

      # The module also checks the email_verified field in the id_token.
      identity_field       = "email"
      identify_with_openid = true
      permitted_identities = ["user@example.com"]
    }
  }
}
```

Limitations
-----------

The Lambda currently does not attempt to deduplicate emails it inserts into Gmail, which helps avoid
granting the Lambda read permissions to your email. If we needed to deduplicate incoming emails,
we'd either have to keep a record of emails we've received somewhere (money + complexity), or we'd
need to search via Gmail's API for an existing copy of the email.

The Lambda does not attempt to handle threading in any sort of clean manner.

The Lambda also does not attempt to pass Gmail's SPF checks; Gmail sees the message as being sent
from some random Amazon, even though it's being uploaded via the Gmail API.

Gmail-to-Gmail, it takes about 15 seconds from hitting send to the email appearing in the
recipient's inbox. This may vary depending on whether the Lambda is warm, and the message size.
