import base64
import json
from os import environ
import re
import time

import boto3
import requests

region = environ.get('AWS_REGION')

s3_client = boto3.client('s3', region_name=region)
ssm_client = boto3.client('ssm', region_name=region)

client_id = environ['GOOGLE_CLIENT_ID']
token_parameter = environ['GOOGLE_TOKEN_PARAMETER']
secret_parameter = environ['GOOGLE_SECRET_PARAMETER']

extra_gmail_label_ids = environ['EXTRA_GMAIL_LABEL_IDS']
base_label_ids = ['INBOX', 'UNREAD']
label_ids = (
    list(set(base_label_ids)
         | set(extra_gmail_label_ids.split(':'))) if extra_gmail_label_ids else base_label_ids
)

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

account_id = environ['AWS_ACCOUNT_ID']


def memoize_dynamic(timeout_fn):
    def inner(fn):
        expiry_mapping = {}

        def reset(*params):
            if params in expiry_mapping:
                del expiry_mapping[params]

        def get(*params):
            prior = expiry_mapping.get(params)
            now = time.monotonic()
            if prior is not None and now < prior[0]:
                return prior[1]
            value = fn(params)
            expiry_mapping[params] = now + timeout_fn(value), value
            return value

        get.reset = reset
        return get

    return inner


def memoize_with_timeout(timeout_sec):
    return memoize_dynamic(lambda _: timeout_sec)


def memoize_with_expiry(grace_period_sec, default_valid_sec):
    return memoize_dynamic(
        lambda value: value.get('expires_in', default_valid_sec) - grace_period_sec
    )


@memoize_with_timeout(timeout_sec=60 * 60)
def get_parameter(parameter_name: str):
    return json.loads(
        ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)['Parameter']['Value']
    )


@memoize_with_expiry(grace_period_sec=5 * 60, default_valid_sec=60 * 60)
def get_access_token():
    client_secret = get_parameter(secret_parameter)['client_secret']
    refresh_token = get_parameter(token_parameter)['refresh_token']
    res = requests.post(
        'https://oauth2.googleapis.com/token',
        data=dict(
            grant_type='refresh_token',
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
        )
    )

    if res.status_code in {401, 403}:
        # Reset parameters.
        get_parameter.reset(secret_parameter)
        get_parameter.reset(token_parameter)

    res.raise_for_status()
    return res.json()


def lambda_handler(event, context):
    message_id = event['Records'][0]['ses']['mail']['messageId']
    print(f'Received message {message_id}')

    object_path = s3_prefix + message_id
    message = s3_client.get_object(
        Bucket=s3_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )['Body'].read()

    token = get_access_token(refresh_token)['access_token']
    auth = f'Bearer {token}'
    # TODO: fix deduplication
    res = requests.post(
        'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
        headers={
            'authorization': auth,
            'content-type': 'message/rfc822',
        },
        data=message,
    )
    if res.status_code in {401, 403}:
        get_access_token.reset()
    if not res.ok:
        raise Exception(f'[{res.status_code}] {res.text}')
    data = res.json()
    id, thread_id = data['id'], data['threadId']
    print(f'Created message {id} in thread {thread_id}')
    res = requests.post(
        f'https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}/modify',
        headers={
            'authorization': auth,
            'content-type': 'application/json',
        },
        data=json.dumps(dict(addLabelIds=label_ids)),
    )
    if res.status_code in {401, 403}:
        get_access_token.reset()
    if not res.ok:
        raise Exception(f'[{res.status_code}] {res.text}')
    print('Updated thread labels')

    s3_client.put_object_tagging(
        Bucket=s3_bucket,
        Key=object_path,
        Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')])
    )
    print('Marked S3 object for deletion')
