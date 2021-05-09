import base64
import json
from os import environ
import time
from typing import Any, Callable, Dict

import boto3
import botocore
import requests

client_id = environ['GOOGLE_CLIENT_ID']
client_secret = environ['GOOGLE_CLIENT_SECRET']
refresh_token = environ['GOOGLE_REFRESH_TOKEN']

extra_gmail_label_ids = environ['EXTRA_GMAIL_LABEL_IDS']
base_label_ids = ['INBOX', 'UNREAD']
label_ids = (
    list(set(base_label_ids) | set(extra_gmail_label_ids.split(':')))
    if extra_gmail_label_ids else base_label_ids
)

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

account_id = environ['AWS_ACCOUNT_ID']


def memoize_with_expiry(grace_period_sec: int, default_valid_sec: int):
    def inner(fn: Callable[[str], Any]) -> Callable[[str], Any]:
        expiry_mapping = {}

        def get(refresh_token: str) -> Any:
            value = expiry_mapping.get(refresh_token)
            now = time.monotonic()
            if value is not None and now < value[0]:
                return value[1]
            value = fn(refresh_token)
            expires_at = now + value.get('expires_in', default_valid_sec) - grace_period_sec
            expiry_mapping[refresh_token] = expires_at, value
            return value

        return get

    return inner


@memoize_with_expiry(grace_period_sec=5 * 60, default_valid_sec=60 * 60)
def get_access_token(refresh_token: str):
    res = requests.post(
        'https://oauth2.googleapis.com/token',
        data=dict(
            grant_type='refresh_token',
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret
        )
    )
    res.raise_for_status()
    return res.json()


def add_length_header_from_response_stream(
        stream: botocore.response.ResponseStream, headers: Dict[str, str]
):
    try:
        headers['content-length'] = int(getattr(stream, '_content_length', None))
    except ValueError:
        pass
    return headers


def lambda_handler(event: Any, context: Any) -> Any:
    message_id = event['Records'][0]['ses']['mail']['messageId']
    print(f'Received message {message_id}')

    client_s3 = boto3.client('s3')

    object_path = s3_prefix + message_id
    message_stream = client_s3.get_object(
        Bucket=s3_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )['Body']

    token = get_access_token(refresh_token)['access_token']
    auth = f'Bearer {token}'
    # TODO: fix deduplication
    res = requests.post(
        'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
        headers=add_length_header_from_response_stream(
            message_stream, {
                'authorization': auth,
                'content-type': 'message/rfc822',
            }
        ),
        data=message_stream,
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
        data=json.dumps(dict(addLabelIds=label_ids)),
    )
    if not res.ok:
        raise Exception(f'[{res.status_code}] {res.text}')
    print('Updated thread labels')

    client_s3.put_object_tagging(
        Bucket=s3_bucket,
        Key=object_path,
        Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')])
    )
    print('Marked S3 object for deletion')
