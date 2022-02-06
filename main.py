from functools import cache
from inspect import getfullargspec, ismethod
import io
import json
from os import environ
from requests_toolbelt import MultipartEncoder
from requests_oauthlib import OAuth2Session
import time
from typing import Callable, Optional, Generic, TypeVar

import boto3
import oauthlib
import requests

region = environ.get('AWS_REGION')

s3_client = boto3.client('s3', region_name=region)
ssm_client = boto3.client('ssm', region_name=region)

client_id = environ['GOOGLE_CLIENT_ID']
token_parameter = environ['GOOGLE_TOKEN_PARAMETER']
secret_parameter = environ['GOOGLE_SECRET_PARAMETER']

extra_gmail_label_ids = environ['EXTRA_GMAIL_LABEL_IDS']
base_label_ids = ['INBOX', 'UNREAD']
label_ids = (list(set(base_label_ids)
                  | set(extra_gmail_label_ids.split(':')))
             if extra_gmail_label_ids else base_label_ids)

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

account_id = environ['AWS_ACCOUNT_ID']


class MultipartRelatedEncoder(MultipartEncoder):

    def _iter_fields(self):
        for field in super()._iter_fields():
            # Gmail's API does not like the content-disposition header, but
            # neither requests not requests_toolbelt have an option to elide it.
            field.headers = {
                h: v
                for h, v in field.headers.items()
                if h.lower() != 'content-disposition'
            }
            yield field

    @property
    def content_type(self):
        return str(f'multipart/related; boundary={self.boundary_value}')


P = TypeVar('P')
T = TypeVar('T')


def memoize_with_timeout(timeout_sec: int):
    """
    Memoize the given function with a TTL. Attempts no cleanup, and thus will
    leak unless used for a fixed set of parameters.
    """

    def inner(fn: Callable[P, T]):
        expiry_mapping: dict[P, tuple[int, T]] = {}

        def get(*params: P) -> T:
            prior = expiry_mapping.get(params)
            now = time.monotonic()
            if prior is not None and now < prior[0]:
                return prior[1]
            value = fn(*params)
            expiry_mapping[params] = now + timeout_sec, value
            return value

        return get

    return inner


@memoize_with_timeout(timeout_sec=60 * 60)
def get_parameter(parameter_name: str):
    return json.loads(
        ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True,
        )['Parameter']['Value'])


def replace_param(fn, args, kwargs, param, value):
    if param in kwargs or not args:
        kwargs[param] = value
        return args
    argspec = getfullargspec(fn)
    try:
        idx = argspec.args.index(param)
    except IndexError:
        idx = None
    if ismethod(fn):
        idx -= 1
    if idx is not None and len(args) > idx:
        return args[:idx] + (value, ) + args[idx + 1:]
    kwargs[param] = value
    return args


class CustomOAuth2Session(OAuth2Session):

    def __init__(self,
                 *args,
                 token_fetcher: Optional[Callable[[], Optional[str]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.token_fetcher = token_fetcher

    def refresh_token(self, *args, **kwargs):
        if not args and 'token_url' not in kwargs:
            kwargs['token_url'] = self.auto_refresh_url
        try:
            return super().refresh_token(*args, **kwargs)
        except (oauthlib.oauth2.rfc6749.errors.InvalidGrantError,
                oauthlib.oauth2.rfc6749.errors.TokenExpiredError):
            if self.token_fetcher is None:
                raise
            new_token = self.token_fetcher()
            if new_token is None or new_token == self.token['refresh_token']:
                raise
            args = replace_param(super().refresh_token, args, kwargs,
                                 'refresh_token', new_token)
            return super().refresh_token(*args, **kwargs)


@cache
def google_session() -> requests.Session:
    client_secret = get_parameter(secret_parameter)['client_secret']
    refresh_token = get_parameter(token_parameter)['refresh_token']
    session = CustomOAuth2Session(
        client_id=client_id,
        auto_refresh_url='https://oauth2.googleapis.com/token',
        auto_refresh_kwargs=dict(
            client_id=client_id,
            client_secret=client_secret,
        ),
        # New tokens that get auto-refreshed should not interrupt the request,
        # as we do not store the new access tokens. This may result in other
        # access tokens expiring randomly, so this only works for low-
        # concurrency applications.
        token_updater=lambda token: None,
        token_fetcher=lambda: get_parameter(token_parameter)['refresh_token'],
        token=dict(
            refresh_token=refresh_token,
            token_type='Bearer',
        ),
    )
    session.refresh_token()
    return session


# N.B. for whatever reason, MultipartEncoder doesn't like doing to_string() with
# this reader. It just spins forever. Possibly because
class SizedReader(object):

    def __init__(self, stream, size: int):
        self.stream = stream
        self.len = size

    def read(self, n=None):
        if n < 0:
            # botocore's StreamingBody will just infinite loop for reads with
            # negative counts, so massage it into "read all."
            n = None
        value = self.stream.read(n)
        self.len = max(0, self.len - len(value))
        assert n is not None or not self.len, f'expected to drain the reader, still have {self.len} bytes'
        return value


def forward_email(message_id: str):
    print(f'Received message {message_id}')

    object_path = s3_prefix + message_id
    obj = s3_client.get_object(
        Bucket=s3_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )

    with google_session() as sess:
        # TODO: fix deduplication
        # TODO: add spam label as appropriate
        encoder = MultipartRelatedEncoder(fields=[
            (None, (None, json.dumps(dict(labelIds=label_ids)),
                    'application/json')),
            (None, (None, SizedReader(obj['Body'], obj['ContentLength']),
                    'message/rfc822')),
        ])
        res = sess.post(
            'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
            headers={'content-type': encoder.content_type},
            data=encoder,
        )
        res.raise_for_status()
        data = res.json()
        gid, thread_id = data['id'], data['threadId']
        print(f'Created message {gid} in thread {thread_id}')

        s3_client.put_object_tagging(
            Bucket=s3_bucket,
            Key=object_path,
            Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')]))
        print('Marked S3 object for deletion')


def lambda_handler(event, context):
    records = event['Records']
    print(f'Processing {len(records)} records')
    for record in records:
        forward_email(record['ses']['mail']['messageId'])
