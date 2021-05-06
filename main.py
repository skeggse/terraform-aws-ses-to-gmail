import base64
import json
from os import environ
import re
import time

import boto3
import requests

client_id = environ['GOOGLE_CLIENT_ID']
client_secret = environ['GOOGLE_CLIENT_SECRET']
refresh_token = environ['GOOGLE_REFRESH_TOKEN']

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

account_id = environ['AWS_ACCOUNT_ID']


def memoize_with_expiry(default_valid_sec, grace_period_sec, field_name):
    def inner(fn):
        expiry_mapping = {}

        def get(refresh_token: str):
            value = expiry_mapping.get(refresh_token)
            now = time.monotonic()
            if value is not None and now < value[0]:
                return value[1]
            value = fn(refresh_token)
            expires_at = now + value.get(field_name,
                                         default_valid_sec) - grace_period_sec
            expiry_mapping[refresh_token] = expires_at, value
            return value

        return get

    return inner


@memoize_with_expiry(
    default_valid_sec=60 * 60,
    grace_period_sec=5 * 60,
    field_name='expires_in')
def get_access_token(refresh_token: str):
    res = requests.post(
        'https://oauth2.googleapis.com/token',
        data=dict(
            grant_type='refresh_token',
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret))
    res.raise_for_status()
    return res.json()


def lambda_handler(event, context):
    message_id = event['Records'][0]['ses']['mail']['messageId']
    print(f'Received message {message_id}')

    client_s3 = boto3.client('s3')

    object_path = incoming_email_prefix + message_id
    message = client_s3.get_object(
        Bucket=incoming_email_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )['Body'].read()

    token = get_access_token(refresh_token)['access_token']
    auth = f'Bearer {token}'
    # TODO: apply a label for better security/visibility.
    # TODO: fix deduplication
    res = requests.post(
        'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
        headers={
            'authorization': auth,
            'content-type': 'message/rfc822',
        },
        data=message,
    )
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
        data=json.dumps(dict(addLabelIds=['INBOX', 'UNREAD'])),
    )
    if not res.ok:
        raise Exception(f'[{res.status_code}] {res.text}')
    print('Updated thread labels')

    client_s3.put_object_tagging(
        Bucket=incoming_email_bucket,
        Key=object_path,
        Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')]))
    print('Marked S3 object for deletion')
