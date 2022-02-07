'''
Forward email blobs stored in S3 (usually delivered by SES) to a private Gmail
inbox via the OAuth2-authorized Gmail API.
'''

from email.message import Message
from email.feedparser import FeedParser
from email.generator import BytesGenerator
from email.parser import BytesParser
from email.utils import format_datetime, parsedate_to_datetime
from functools import cache, reduce
from inspect import getfullargspec, ismethod
import io
import json
from os import environ
from requests_toolbelt import MultipartEncoder
from requests_oauthlib import OAuth2Session
import time
from typing import Callable, Iterable, Optional, TypeVar

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
label_ids = (
    list(set(base_label_ids) | set(extra_gmail_label_ids.split(':')))
    if extra_gmail_label_ids
    else base_label_ids
)

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

account_id = environ['AWS_ACCOUNT_ID']


class MultipartRelatedEncoder(MultipartEncoder):
    '''
    An extension to `MultipartEncoder` which enables encoding the content as
    `multipart/related` instead of `multipart/form-data`, which includes the
    elision of the `content-disposition` header.
    '''

    def _iter_fields(self):
        for field in super()._iter_fields():
            # Gmail's API does not like the content-disposition header, but
            # neither requests nor requests_toolbelt have an option to elide it.
            field.headers = {
                h: v for h, v in field.headers.items() if h.lower() != 'content-disposition'
            }
            yield field

    @property
    def content_type(self):
        return f'multipart/related; boundary={self.boundary_value}'


P = TypeVar('P')
T = TypeVar('T')


def memoize_with_timeout(timeout_sec: int):
    '''
    Memoize the given function with a TTL. Attempts no cleanup, and thus will
    leak unless used for a fixed set of parameters.
    '''

    def inner(func: Callable[P, T]):
        expiry_mapping: dict[P, tuple[int, T]] = {}

        def get(*params: P) -> T:
            prior = expiry_mapping.get(params)
            now = time.monotonic()
            if prior is not None and now < prior[0]:
                return prior[1]
            value = func(*params)
            expiry_mapping[params] = now + timeout_sec, value
            return value

        return get

    return inner


@memoize_with_timeout(timeout_sec=60 * 60)
def get_parameter(parameter_name: str):
    '''
    Retrieve the SSM parameter with the given name, and cache it for an hour.
    '''
    return json.loads(
        ssm_client.get_parameter(Name=parameter_name, WithDecryption=True,)[
            'Parameter'
        ]['Value']
    )


def replace_param(func, args, kwargs, param, value):
    '''
    Given a function and a set of positional and keyword arguments for that
    function, replace or add the parameter with the given `param` name and give
    it the given `value`.
    '''

    if param in kwargs or not args:
        kwargs[param] = value
        return args
    argspec = getfullargspec(func)
    try:
        idx = argspec.args.index(param)
    except IndexError:
        idx = None
    if ismethod(func):
        idx -= 1
    if idx is not None and len(args) > idx:
        return args[:idx] + (value,) + args[idx + 1 :]
    kwargs[param] = value
    return args


class CustomOAuth2Session(OAuth2Session):
    '''
    An extension of `OAuth2Session` that supports pulling the refresh token from
    another source when the token appears to be expired or otherwise invalid.
    '''

    def __init__(
        self, *args, token_fetcher: Optional[Callable[[], Optional[str]]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.token_fetcher = token_fetcher

    def refresh_token(self, *args, **kwargs):
        if not args and 'token_url' not in kwargs:
            kwargs['token_url'] = self.auto_refresh_url
        try:
            return super().refresh_token(*args, **kwargs)
        except (
            oauthlib.oauth2.rfc6749.errors.InvalidGrantError,
            oauthlib.oauth2.rfc6749.errors.TokenExpiredError,
        ):
            if self.token_fetcher is None:
                raise
            new_token = self.token_fetcher()
            if new_token is None or new_token == self.token['refresh_token']:
                raise
            args = replace_param(super().refresh_token, args, kwargs, 'refresh_token', new_token)
            return super().refresh_token(*args, **kwargs)


@cache
def google_session() -> requests.Session:
    '''
    Produce a multi-use Session object authenticated for the Google API. Take
    care not to pass the access token to a non-Google API - this does not
    carefully ensure the token is only passed to https://gmail.googleapis.com.
    '''

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


class GmailMessage:
    '''
    Represents a single message object in Gmail, and populates its fields from
    the API. Only provides metadata.
    '''

    def __init__(self, message_id, thread_id=None):
        self.message_id = message_id
        self._thread_id = thread_id
        self._history_id = None
        self._metadata = None
        self._headers = None
        self._internal_date = None
        self._rfc822_message_id = None

    def _load(self):
        with google_session() as sess:
            res = sess.get(
                f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{self.message_id}',
                params={
                    'format': 'metadata',
                    'metadataHeaders': ['internalDate', 'message-id', 'x-ses-receipt'],
                },
            )
            res.raise_for_status()
            metadata = res.json()
            self._metadata = metadata
            self._thread_id = metadata['threadId']
            self._history_id = metadata['historyId']
            self._headers = metadata['payload']['headers']
            self._internal_date = metadata['internalDate']

            self._rfc822_message_id = self['message-id']

    @property
    def thread_id(self):
        if self._thread_id is None:
            self._load()
        return self._thread_id

    @property
    def history_id(self):
        if self._history_id is None:
            self._load()
        return self._history_id

    @property
    def rfc822_message_id(self):
        if self._rfc822_message_id is None:
            self._load()
        return self._rfc822_message_id

    @property
    def headers(self):
        if self._headers is None:
            self._load()
        return self._headers

    @property
    def internal_date(self):
        if self._internal_date is None:
            self._load()
        return self._internal_date

    def api(self, action: str = ''):
        '''
        Construct the API URL for the message, along with the given `action`.
        '''
        if action and not action.startswith('/'):
            action = f'/{action}'
        return f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{self.message_id}{action}'

    def __getitem__(self, header: str):
        '''
        Get the value of the single instance of the given `header`. If there are
        multiple such values or no such values, this will fail.
        '''
        value = self.get(header)
        if value is None:
            raise KeyError(f'no header for {header}')
        return value

    def get(self, header: str, default: Optional[str] = None) -> Optional[str]:
        '''
        Get the value of the single instance of the given `header`, if any. If
        there are multiple such values, this will fail. If there are no such
        values, this will return the `default` value.
        '''
        values = list(self.get_all(header))
        assert len(values) <= 1, f'got multiple headers for {header}'
        return values[0] if values else default

    def get_all(self, header: str) -> Iterable[str]:
        '''
        Get all values for the given `header`.
        '''
        lower_header = header.lower()
        return (entry['value'] for entry in self.headers if entry['name'].lower() == lower_header)


def prune_message(m1: GmailMessage, m2: GmailMessage):
    '''
    Given two supposedly identical messages, prune the one that seems to have
    been inserted most recently.
    '''

    assert m1.message_id != m2.message_id, 'Pruning one of two identical messages'
    assert (
        m1['x-ses-receipt'] == m2['x-ses-receipt']
    ), 'Refusing to merge duplicate messages that have different receipt IDs'
    message_to_keep, message_to_remove = sorted(
        (m1, m2), key=lambda m: (m.internal_date, m.history_id, m.message_id)
    )

    with google_session() as sess:
        sess.post(message_to_remove.api('/trash'))
    return message_to_keep


def list_emails_by_rfc822_msg_id(rfc822_msg_id: str) -> Iterable[GmailMessage]:
    '''
    Enumerate Gmail messages with the given rfc822 Message-ID header as
    GmailMessage objects.
    '''
    with google_session() as sess:
        res = sess.get(
            'https://gmail.googleapis.com/gmail/v1/users/me/messages',
            params=dict(q=f'rfc822msgid:{rfc822_msg_id}'),
        )
        res.raise_for_status()
        data = res.json()
        for message in data.get('messages', ()):
            gmsg = GmailMessage(message['id'], message['threadId'])
            msg_id = gmsg.get('Message-ID')
            assert (
                msg_id is not None
            ), f'no Message-ID found for {gmsg.message_id} ({rfc822_msg_id})'
            if msg_id == rfc822_msg_id:
                yield gmsg
            else:
                # Suggests malicious behavior. TODO: route this somewhere that
                # alerts.
                print(f'matched `{msg_id}` == `{rfc822_msg_id}` for {gmsg.message_id}')


def deduplicate_email(rfc822_msg_id: str, ses_id: Optional[str] = None) -> int:
    '''
    Deduplicate emails with the given rfc822 Message-ID header. Optionally
    ensure that they all use the given X-SES-Receipt header as a further safety
    belt.
    '''
    messages = list(list_emails_by_rfc822_msg_id(rfc822_msg_id))
    if ses_id is not None:
        assert all(
            m['x-ses-receipt'] == ses_id for m in messages
        ), 'Multiple SES receipts for the same rfc822 message ID'
    if 1 < len(messages) <= 3:
        kept_message = reduce(prune_message, messages)
        print(f'Kept {kept_message.message_id} for {rfc822_msg_id}')
    return len(messages)


assert callable(BytesGenerator._dispatch), 'private BytesGenerator contract has changed'


class RawBytesGenerator(BytesGenerator):
    '''
    Generate a valid rfc822-encoded message that was decoded using the
    headersonly=True parameter using `email.parser`. This ensures we don't
    change the line-ending style and don't attempt to re-encode the bytes
    themselves.
    '''

    def _write(self, msg: Message):
        self._write_headers(msg)
        # Bypass all the serialization and line conversions.
        self.write(msg.get_payload())

    @staticmethod
    def convert_bytes(msg: Message, linesep: str = '\r\n') -> bytes:
        '''
        Serialize the given message into a buffer, and return the corresponding
        `bytes`.
        '''
        fp = io.BytesIO()
        g = RawBytesGenerator(fp, mangle_from_=False, policy=msg.policy.clone(max_line_length=None))
        g.flatten(msg, linesep=linesep)
        return fp.getvalue()


def infer_linesep(data: bytes, default: str = '\n') -> str:
    r'''
    Infer the line separator from the given data. Defaults to `'\n'` if the data
    contains no line separator.
    '''
    try:
        idx = data.index(b'\n')
    except IndexError:
        return default
    return '\r\n' if idx > 0 and data[idx - 1] == b'\r'[0] else '\n'


def forward_email(message_id: str, proactive_duplicate_check: bool = False):
    '''
    Forward the email with the given SES message ID, corresponding to the S3
    object key. Optionally verify that the email has not already been forwarded
    when `proactive_duplicate_check=True`.
    '''

    print(f'Received message {message_id}')

    object_path = s3_prefix + message_id
    obj = s3_client.get_object(
        Bucket=s3_bucket,
        Key=object_path,
        ExpectedBucketOwner=account_id,
    )

    msg_bytes = obj['Body'].read()
    pmsg = BytesParser().parsebytes(msg_bytes, headersonly=True)

    mark_as_spam = (
        pmsg.get('x-ses-spam-verdict') != 'PASS' or pmsg.get('x-ses-virus-verdict') != 'PASS'
    )
    rfc822_msg_id = pmsg['message-id']

    if proactive_duplicate_check:
        messages_matched = deduplicate_email(rfc822_msg_id, ses_id=pmsg['x-ses-receipt'])
        if messages_matched:
            print(f'Proactive duplicate check matched {messages_matched} messages')
            return

    msg_label_ids = label_ids + ['SPAM'] if mark_as_spam else label_ids

    obj_date = obj['LastModified']

    message_metadata = dict(labelIds=msg_label_ids)
    date_header = pmsg.get('date')
    parsed_date = date_header and parsedate_to_datetime(date_header)
    update_timestamp = (
        not parsed_date or abs(parsed_date.timestamp() - obj_date.timestamp()) > 5 * 60
    )
    if update_timestamp:
        new_header = format_datetime(
            obj_date.astimezone(parsed_date.tzinfo) if parsed_date else obj_date
        )
        print(f'Overwriting date header: {new_header}')
        # TODO: internalDate instead? Can't seem to get it to work.
        if 'date' in pmsg:
            pmsg.replace_header('Date', new_header)
        else:
            pmsg['Date'] = new_header
        # TODO: stream this instead of buffering it.
        msg_bytes = RawBytesGenerator.convert_bytes(pmsg, linesep=infer_linesep(msg_bytes))

    with google_session() as sess:
        # TODO: handle message threading
        # TODO: can we get eventbridge to tell us if this is a possible event
        # redelivery?
        encoder = MultipartRelatedEncoder(
            fields=[
                (None, (None, json.dumps(message_metadata), 'application/json')),
                (None, (None, msg_bytes, 'message/rfc822')),
            ]
        )
        res = sess.post(
            'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
            headers={'content-type': encoder.content_type},
            data=encoder,
            params=dict(internalDateSource='dateHeader'),
        )
        res.raise_for_status()
        data = res.json()
        gid, thread_id = data['id'], data['threadId']
        print(f'Created message {gid} in thread {thread_id}')

        s3_client.put_object_tagging(
            Bucket=s3_bucket,
            Key=object_path,
            Tagging=dict(TagSet=[dict(Key='Forwarded', Value='true')]),
        )
        print('Marked S3 object for deletion')

        matched_emails = deduplicate_email(rfc822_msg_id)
        if not matched_emails:
            print(f'No messages found for rfc822 message id `{rfc822_msg_id}`')
        elif matched_emails > 3:
            # TODO: this should really notify
            print(
                'Tried to deduplicate more than three messages for rfc822 '
                f'message id `{rfc822_msg_id}`!'
            )


def lambda_handler(event, context):
    '''
    Handle the Lambda invocation, either via SES -> SNS, or via manual
    invocation to handle operational outages/authentication failure gaps.
    '''

    records = event.get('Records')
    if not records:
        # Assume we need to comb through the S3 bucket for recovery purposes.
        for page in s3_client.get_paginator('list_objects_v2').paginate(
            Bucket=s3_bucket, ExpectedBucketOwner=account_id
        ):
            for obj in page['Contents']:
                key = obj['Key']
                do_forward = (
                    True
                    if event.get('ignoreTags', False)
                    else not next(
                        (
                            entry['Value'] == 'true'
                            for entry in s3_client.get_object_tagging(
                                Bucket=s3_bucket, Key=key, ExpectedBucketOwner=account_id
                            )['TagSet']
                        ),
                        False,
                    )
                )
                if do_forward:
                    # TODO: don't re-forward deleted emails.
                    forward_email(key, proactive_duplicate_check=True)
    else:
        print(f'Processing {len(records)} records')
        for record in records:
            forward_email(record['ses']['mail']['messageId'])
