from email.parser import BytesParser
from email.feedparser import FeedParser
from email.utils import format_datetime, parsedate_to_datetime
from functools import cache, reduce
from inspect import getfullargspec, ismethod
import io
import json
from os import environ
from requests_toolbelt import MultipartEncoder
from requests_oauthlib import OAuth2Session
import time
from typing import Callable, Optional, TypeVar

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


class SpecialReader(io.RawIOBase):

    def __init__(self, stream):
        self.stream = stream
        self._prev_byte = None
        self.line_ending = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def readable(self):
        return True

    def read(self, n=None):
        # botocore's StreamingBody will just infinite loop for reads with
        # negative counts, so massage it into "read all."
        value = self.stream.read(None if n < 0 else n)
        if self.line_ending is None:
            try:
                idx = value.index(b'\n')
            except IndexError:
                print('not in first block')
                self._prev_byte = value[-1]
                return value
            if idx != 0: self._prev_byte = value[idx - 1]
            self.line_ending = '\r\n' if self._prev_byte == b'\r'[0] else '\n'
        return value


def get_message_metadata(message):
    with google_session() as sess:
        res = sess.get(
            f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{message["id"]}',
            params={
                'format': 'metadata',
                'metadataHeaders': ['internalDate', 'message-id']
            })
        res.raise_for_status()
        return res.json()


def prune_message(m1, m2):
    message_to_keep, message_to_remove = sorted(
        (m1, m2), key=lambda m: (m['internalDate'], m['historyId'], m['id']))
    assert message_to_keep['id'] != message_to_remove[
        'id'], 'Pruning one of two identical messages'

    with google_session() as sess:
        sess.post(
            f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_to_remove["id"]}/trash'
        )
    return message_to_keep


def list_emails_by_rfc822_msg_id(rfc822_msg_id: str):
    with google_session() as sess:
        res = sess.get(
            'https://gmail.googleapis.com/gmail/v1/users/me/messages',
            params=dict(q=f'rfc822msgid:{rfc822_msg_id}'))
        res.raise_for_status()
        data = res.json()
        for message in data.get('messages', ()):
            message = get_message_metadata(message)
            msg_id = next((entry['value']
                           for entry in message['payload']['headers']
                           if entry['name'].lower() == 'message-id'), None)
            assert msg_id is not None, f'no Message-ID found for {message["id"]} ({rfc822_msg_id})'
            if msg_id == rfc822_msg_id:
                yield message
            else:
                # Suggests malicious behavior. TODO: route this somewhere that
                # alerts.
                print(
                    f'matched `{msg_id}` == `{rfc822_msg_id}` for {message["id"]}'
                )


# TODO: also ensure the x-ses-receipt header matches before removing any emails.
def deduplicate_email(rfc822_msg_id: str) -> int:
    with google_session() as sess:
        messages = list(list_emails_by_rfc822_msg_id(rfc822_msg_id))
        if 1 < len(messages) <= 3:
            kept_message = reduce(prune_message, messages)
            print(f'Kept {kept_message["id"]} for {rfc822_msg_id}')
        return len(messages)


def infer_linesep(data: bytes) -> str:
    return '\r\n' if data[data.index(b'\n') - 1] == b'\r'[0] else '\n'


def pipe(reader, dest):
    while True:
        data = reader.read(io.DEFAULT_BUFFER_SIZE)
        if not data: break
        dest(data)


def parse_email_headers(reader):
    feedparser = FeedParser()
    feedparser._set_headersonly()
    fp = io.TextIOWrapper(reader,
                          encoding='ascii',
                          errors='surrogateescape',
                          newline='')
    # TODO: stop feeding once headers have been parsed.
    pipe(fp, feedparser.feed)
    return feedparser.close()


def forward_email(message_id: str, proactive_duplicate_check: bool = False):
    print(f'Received message {message_id}')

    object_path = s3_prefix + message_id
    obj = s3_client.get_object(
        Bucket=s3_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )

    # TODO: tee the read stream so we can pass it through to the multipart
    # encoder.
    with SpecialReader(obj['Body']) as reader:
        pmsg = parse_email_headers(reader)

    mark_as_spam = pmsg.get('x-ses-spam-verdict') != 'PASS' or pmsg.get(
        'x-ses-virus-verdict') != 'PASS'
    rfc822_msg_id = pmsg['message-id']

    if proactive_duplicate_check:
        messages_matched = deduplicate_email(rfc822_msg_id)
        if messages_matched:
            print(
                f'Proactive duplicate check matched {messages_matched} messages'
            )
            return

    msg_label_ids = label_ids + ['SPAM'] if mark_as_spam else label_ids

    obj_date = obj['LastModified']

    message_metadata = dict(labelIds=msg_label_ids)
    date_header = pmsg.get('date')
    parsed_date = date_header and parsedate_to_datetime(date_header)
    update_timestamp = (
        not parsed_date
        or abs(parsed_date.timestamp() - obj_date.timestamp()) > 5 * 60)
    if update_timestamp:
        message_metadata['internalDate'] = int(obj_date.timestamp() * 1000)

    with google_session() as sess:
        # TODO: fix deduplication
        # TODO: handle threading
        # TODO: can we get eventbridge to tell us if this is a possible event
        # redelivery?
        encoder = MultipartRelatedEncoder(fields=[
            # TODO: internalDate instead (64bit ms since epoch)?
            (None, (None, json.dumps(message_metadata), 'application/json')),
            (None, (None, msg_bytes, 'message/rfc822')),
        ])
        res = sess.post(
            'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
            headers={'content-type': encoder.content_type},
            data=encoder,
            params=dict(internalDateSource='dateHeader'))
        res.raise_for_status()
        data = res.json()
        gid, thread_id = data['id'], data['threadId']
        print(f'Created message {gid} in thread {thread_id}')

        s3_client.put_object_tagging(
            Bucket=s3_bucket,
            Key=object_path,
            Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')]))
        print('Marked S3 object for deletion')

        matched_emails = deduplicate_email(rfc822_msg_id)
        if not matched_emails:
            print(f'No messages found for rfc822 message id `{rfc822_msg_id}`')
        elif matched_emails > 3:
            # TODO: this should really notify
            print(
                f'Tried to deduplicate more than three messages for rfc822 message id `{rfc822_msg_id}`!'
            )


def lambda_handler(event, context):
    records = event['Records']
    print(f'Processing {len(records)} records')
    for record in records:
        forward_email(record['ses']['mail']['messageId'])
