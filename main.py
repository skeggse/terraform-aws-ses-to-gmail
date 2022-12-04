'''
Forward email blobs stored in S3 (usually delivered by SES) to a private Gmail
inbox via the OAuth2-authorized Gmail API.
'''

from datetime import datetime, timedelta, timezone
from email.message import Message
from email.generator import BytesGenerator
from email.parser import BytesParser
from email.utils import format_datetime, parsedate_to_datetime, getaddresses
from enum import Enum
from functools import cache, partial, wraps
from inspect import getfullargspec, ismethod
import io
import json
from os import environ
import time
from typing import Any, Callable, Iterable, Optional, TypedDict, TypeVar, Union
import weakref

import boto3
from botocore.response import StreamingBody
import oauthlib
import requests
from urllib3.fields import RequestField
from requests_oauthlib import OAuth2Session
from requests_toolbelt import MultipartEncoder

region = environ.get('AWS_REGION')

# Window to allow a Lambda to attempt to delivery a message to Gmail. If the start timestamp on the
# message is more than this old, then we may re-attempt delivery.
GMAIL_DELIVERY_TIMEOUT = timedelta(minutes=2)

# The acceptable number of seconds it may take for a message to be delivered, as measured by the
# difference between the S3 object timestamp and the RFC822 Date header. Anecdotally, I've seen a
# real message take seven minutes to get delivered.
ACCEPTABLE_DELIVERY_DELAY = 15 * 60

s3_client = boto3.client('s3', region_name=region)
sns_client = boto3.client('sns', region_name=region)
ssm_client = boto3.client('ssm', region_name=region)

client_id = environ['GOOGLE_CLIENT_ID']
token_parameter = environ['GOOGLE_TOKEN_PARAMETER']
secret_parameter = environ['GOOGLE_SECRET_PARAMETER']

extra_gmail_label_ids = environ['EXTRA_GMAIL_LABEL_IDS']
if extra_gmail_label_ids:
    extra_gmail_label_ids = set(extra_gmail_label_ids.split(':'))
base_label_ids = ['UNREAD']  # INBOX/SPAM get added conditionally.
label_ids = list(set(base_label_ids) | extra_gmail_label_ids)

s3_bucket = environ['S3_BUCKET']
s3_prefix = environ.get('S3_PREFIX', '')

EVENTS_TOPIC_ARN = environ.get('EVENTS_TOPIC_ARN')

account_id = environ['AWS_ACCOUNT_ID']


T = TypeVar('T')  # pylint: disable=invalid-name


def cachedmethod(func: T) -> T:
    '''
    Cache a method associated with specific objects, such that the cache is cleaned up along with
    the object that the method is attached to. This avoids the memory leaks associated with just
    using @cache on a regular method.
    '''

    cache_dict = weakref.WeakKeyDictionary()

    @wraps(func)
    def method(self, *args, **kwargs):
        instance_cache = cache_dict.get(self)
        if instance_cache is None:
            instance_cache = cache(partial(func, self))
            cache_dict[self] = instance_cache
        return instance_cache(*args, **kwargs)

    return method


def cachedproperty(func):
    '''Cache a dynamic property for a class, avoiding the potential memory leak.'''

    return property(cachedmethod(func))


def args_key(args, kwargs):
    '''Get a hashable representation of the args and kwargs from a function invocation.'''

    return (args, frozenset(kwargs.items()))


P = TypeVar('P')  # pylint: disable=invalid-name


def memoize_with_timeout(timeout_sec: int):
    '''
    Memoize the given function with a TTL. Attempts no cleanup, and thus will leak unless used for a
    fixed set of parameters.
    '''

    def inner(func):
        expiry_mapping: dict[P, tuple[float, T]] = {}

        def get(*params, **kwargs) -> T:
            key = args_key(params, kwargs)
            prior = expiry_mapping.get(key)
            now = time.monotonic()
            if prior is not None and now < prior[0]:
                return prior[1]
            value = func(*params, **kwargs)
            expiry_mapping[key] = now + timeout_sec, value
            return value

        return get

    return inner


def replace_param(func, args, kwargs, param, value):
    '''
    Given a function and a set of positional and keyword arguments for that function, replace or add
    the parameter with the given `param` name and give it the given `value`.
    '''

    if param in kwargs or not args:
        kwargs[param] = value
        return args
    argspec = getfullargspec(func)
    try:
        idx = argspec.args.index(param) - ismethod(func)
    except IndexError:
        idx = None
    if idx is not None and len(args) > idx:
        return args[:idx] + (value,) + args[idx + 1 :]
    kwargs[param] = value
    return args


@memoize_with_timeout(timeout_sec=60 * 60)
def get_parameter(parameter_name: str):
    '''
    Retrieve the SSM parameter with the given name, and cache it for an hour.
    '''
    return json.loads(
        ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)['Parameter']['Value']
    )


class MultipartRelatedEncoder(MultipartEncoder):
    '''
    An extension to `MultipartEncoder` which enables encoding the content as `multipart/related`
    instead of `multipart/form-data`, which includes the elision of the `content-disposition`
    header.
    '''

    def _iter_fields(self) -> Iterable[RequestField]:
        for field in super()._iter_fields():
            # Gmail's API does not like the content-disposition header, but neither requests nor
            # requests_toolbelt have an option to elide it.
            field.headers = {
                h: v for h, v in field.headers.items() if h.lower() != 'content-disposition'
            }
            yield field

    @property
    def content_type(self) -> str:
        return f'multipart/related; boundary={self.boundary_value}'


class CustomOAuth2Session(OAuth2Session):
    '''
    An extension of `OAuth2Session` that supports pulling the refresh token from another source when
    the token appears to be expired or otherwise invalid.
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
            # This may mutate kwargs in-place.
            args = replace_param(super().refresh_token, args, kwargs, 'refresh_token', new_token)
            return super().refresh_token(*args, **kwargs)


@cache
def google_session() -> requests.Session:
    '''
    Produce a multi-use Session object authenticated for the Google API. Take care not to pass the
    access token to a non-Google API - this does not carefully ensure the token is only passed to
    https://gmail.googleapis.com.
    '''

    client_secret: str = get_parameter(secret_parameter)['client_secret']
    refresh_token: str = get_parameter(token_parameter)['refresh_token']
    session = CustomOAuth2Session(
        client_id=client_id,
        auto_refresh_url='https://oauth2.googleapis.com/token',
        auto_refresh_kwargs=dict(
            client_id=client_id,
            client_secret=client_secret,
        ),
        # New tokens that get auto-refreshed should not interrupt the request, as we do not store
        # the new access tokens. This may result in other access tokens expiring randomly, so this
        # only works for low- concurrency applications.
        token_updater=lambda token: None,
        token_fetcher=lambda: get_parameter(token_parameter)['refresh_token'],
        token=dict(
            refresh_token=refresh_token,
            token_type='Bearer',
        ),
    )
    session.refresh_token()
    return session


class GmailHeader(TypedDict):
    '''A single Gmail message header pair.'''

    name: str
    value: str


class GmailMessage:
    '''
    Represents a single message object in Gmail, and populates its fields from the API. Only
    provides metadata.
    '''

    def __init__(self, message_id: str, thread_id: Optional[str] = None):
        self.message_id = message_id
        self._thread_id = thread_id

    @cachedproperty
    def metadata(self) -> dict[str, Any]:
        '''
        Fetch (and cache) the message's metadata.
        '''
        with google_session() as sess:
            res = sess.get(
                f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{self.message_id}',
                params=dict(
                    format='metadata',
                    metadataHeaders=['internalDate', 'message-id', 'x-ses-receipt'],
                ),
                timeout=10,
            )
            res.raise_for_status()
            return res.json()

    @property
    def thread_id(self) -> str:
        '''The (cached) thread ID.'''
        return self._thread_id or self.metadata['threadId']

    @property
    def history_id(self) -> str:
        '''The (cached) history ID.'''
        return self.metadata['historyId']

    @property
    def rfc822_message_id(self) -> str:
        '''The (cached) rfc822 Message-ID.'''
        return self['message-id']

    @property
    def headers(self) -> list[GmailHeader]:
        '''The (cached) message headers.'''
        return self.metadata['payload']['headers']

    @property
    def internal_date(self) -> int:
        '''The (cached) internal date timestamp, as milliseconds since the UTC UNIX epoch.'''
        return int(self.metadata['internalDate'])

    @property
    def label_ids(self) -> list[str]:
        '''The (cached) list of label IDs associated with the message.'''
        return self.metadata['labelIds']

    def api(self, action: str = '') -> str:
        '''
        Construct the API URL for the message, along with the given `action`.
        '''
        if action and not action.startswith('/'):
            action = f'/{action}'
        return f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{self.message_id}{action}'

    def __getitem__(self, header: str) -> str:
        '''
        Get the value of the single instance of the given `header`. If there are multiple such
        values or no such values, this will fail.
        '''
        value = self.get(header)
        if value is None:
            raise KeyError(f'no header for {header}')
        return value

    def get(self, header: str, default: Optional[str] = None) -> Optional[str]:
        '''
        Get the value of the single instance of the given `header`, if any. If there are multiple
        such values, this will fail. If there are no such values, this will return the `default`
        value.
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


def list_emails_by_rfc822_msg_id(rfc822_msg_id: str) -> Iterable[GmailMessage]:
    '''
    Enumerate Gmail messages with the given rfc822 Message-ID header as GmailMessage objects. Does
    not implement pagination.
    '''
    with google_session() as sess:
        # Note that we don't trust the RFC822 Message-ID header, because it's arbitrary user input
        # from external actors. This could result in a far more expansive query (if it managed to
        # include a space), or a query that overlaps with a different message. Thus, we must cross-
        # check the RFC822 Message-ID header of the matched messages, and must verify that they
        # correspond to the same SES S3 object.
        res = sess.get(
            'https://gmail.googleapis.com/gmail/v1/users/me/messages',
            params=dict(q=f'rfc822msgid:{rfc822_msg_id}', includeSpamTrash='true'),
            timeout=30,
        )
        res.raise_for_status()
        data = res.json()
        for message in data.get('messages', ()):
            gmsg = GmailMessage(message['id'], message['threadId'])
            # Ignore trashed messages for the purposes of deduplication.
            if 'TRASH' in gmsg.label_ids:
                continue
            msg_id = gmsg.get('Message-ID')
            assert (
                msg_id is not None
            ), f'no Message-ID found for {gmsg.message_id} ({rfc822_msg_id})'
            if msg_id == rfc822_msg_id:
                yield gmsg
            else:
                # Suggests malicious behavior. TODO: route this somewhere that alerts.
                print(f'matched `{msg_id}` == `{rfc822_msg_id}` for {gmsg.message_id}')


def deduplicate_email(rfc822_msg_id: str, ses_id: Optional[str] = None) -> int:
    '''
    Deduplicate emails with the given rfc822 Message-ID header. Optionally ensure that they all use
    the given X-SES-Receipt header as a further safety belt.
    '''
    messages = list(list_emails_by_rfc822_msg_id(rfc822_msg_id))
    all_ses_ids = {
        m['x-ses-receipt']
        for m in messages
        # Messages (usually test messages) sent from the same Gmail account will show up multiple
        # times; this isn't helpful, because they won't have the x-ses-receipt header. Skip matching
        # messages that aren't in all the expected extra labels.
        if all(label in m.label_ids for label in extra_gmail_label_ids)
    }
    if ses_id is not None:
        all_ses_ids.add(ses_id)
    assert all_ses_ids, 'No SES receipts for the inserted message'
    assert len(all_ses_ids) == 1, 'Multiple SES receipts for the same rfc822 message ID'
    if 1 < len(messages) <= 3:
        kept_message = min(messages, key=lambda m: (m.internal_date, m.history_id, m.message_id))
        with google_session() as sess:
            for message_to_remove in messages:
                if message_to_remove is kept_message:
                    continue
                assert message_to_remove.message_id != kept_message.message_id
                sess.post(message_to_remove.api('/trash'), timeout=10)
        print(f'Kept {kept_message.message_id} for {rfc822_msg_id}')
    return len(messages)


assert callable(getattr(BytesGenerator, '_write', None)) and callable(
    getattr(BytesGenerator, '_write_headers', None)
), 'private BytesGenerator contract has changed'


class RawBytesGenerator(BytesGenerator):
    '''
    Generate a valid rfc822-encoded message that was decoded using the headersonly=True parameter
    using `email.parser`. This ensures we don't change the line-ending style and don't attempt to
    re-encode the bytes themselves.
    '''

    def _write(self, msg: Message) -> None:
        self._write_headers(msg)

        # These message objects are not multipart because we only parsed the headers. The
        # get_payload method's behavior is substantially different if this invariant is violated.
        assert not msg.is_multipart()

        # Bypass all the serialization and line conversions.
        #
        # Write the raw payload directly, rather than involving get_payload, because the latter
        # attempts various forms of decoding despite the fact that we set headers_only to True.
        # This results in occasional failed attempts at re-encoding the data, which is a mistaken
        # to begin with.
        self.write(msg._payload)

    @staticmethod
    def convert_bytes(msg: Message, linesep: str = '\r\n') -> bytes:
        '''
        Serialize the given message into a buffer, and return the corresponding `bytes`.
        '''
        bytes_io = io.BytesIO()
        RawBytesGenerator(
            bytes_io, mangle_from_=False, policy=msg.policy.clone(max_line_length=None)
        ).flatten(msg, linesep=linesep)
        return bytes_io.getvalue()


def infer_linesep(data: bytes, default: str = '\n') -> str:
    r'''Infer the line separator from the given data. Defaults to `'\n'` if the data contains no
    line separator.
    '''
    try:
        idx = data.index(b'\n')
    except IndexError:
        return default
    return '\r\n' if idx > 0 and data[idx - 1] == b'\r'[0] else '\n'


def get_object_tag(*, bucket: str = s3_bucket, key: str, tag: str) -> Optional[str]:
    '''
    Get the given case-sensitive `tag` for the given `key`, None if no such tag exists.
    '''
    return next(
        (
            entry['Value']
            for entry in s3_client.get_object_tagging(
                Bucket=bucket, Key=key, ExpectedBucketOwner=account_id
            )['TagSet']
            if entry['Key'] == tag
        ),
        None,
    )


class S3Object(TypedDict, total=False):
    '''The response from a get_object S3 API call.'''

    AcceptRanges: str
    Body: StreamingBody
    ContentLength: int
    ContentType: str
    ETag: str
    LastModified: datetime
    Metadata: dict[str, Any]
    ResponseMetadata: Any
    ServerSideEncryption: Union[Any, str]
    TagCount: int


class SESMessage:
    '''
    A representation of an SES message that's been delivered to S3, along with some parsed metadata.
    '''

    _buffer: Optional[bytes]

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self._buffer = None

    @property
    def buffer(self) -> bytes:
        '''
        Get the current message buffer, which may or may not have been modified from the S3
        representation to support clarifications.
        '''
        return self._buffer if self._buffer is not None else bytes(self)

    @buffer.setter
    def buffer(self, value: bytes) -> None:
        '''
        Set the SES message's buffer, to support modifying the buffer with clarifications -
        particularly as a result of delayed delivery.
        '''
        self._buffer = value

    @cachedproperty
    def obj(self) -> S3Object:
        '''
        The (cached) object representation from S3, including a cached body. Prefer bytes(self) to
        get access to the body content, as this stream is only provided once and may be consumed.
        '''
        return s3_client.get_object(
            Bucket=self.bucket, Key=self.key, ExpectedBucketOwner=account_id
        )

    @cachedmethod
    def __bytes__(self) -> bytes:
        return self.body.read()

    @cachedmethod
    def email_message(self) -> Message:
        '''
        The (cached) parsed `email.message.Message` object from the raw object content. Only
        includes parsed headers, and does not include a parsed representation of the body. The
        body's raw encoded content is available through the Message's `get_payload` method.
        '''
        return BytesParser().parsebytes(bytes(self), headersonly=True)

    @property
    def rfc822_message_id(self) -> str:
        '''The message's (cached) rfc822 Message-ID header'''
        return self.email_message()['message-id']

    @cachedmethod
    def get_tags(self) -> dict[str, str]:
        '''The message's current S3 object tags, as a dictionary.'''
        return {
            entry['Key']: entry['Value']
            for entry in s3_client.get_object_tagging(
                Bucket=self.bucket, Key=self.key, ExpectedBucketOwner=account_id
            )['TagSet']
        }

    def was_forwarded(self) -> bool:
        '''Check whether the message has already been marked as forwarded. Cached.'''
        return self.get_tags().get('Forwarded') == 'true'

    def set_tags(self, tags: dict[str, Optional[str]]) -> None:
        '''Mark the message's S3 object with particular tags.'''
        s3_client.put_object_tagging(
            Bucket=self.bucket,
            Key=self.key,
            Tagging=dict(
                TagSet=[
                    dict(Key=key, Value=value) for key, value in tags.items() if value is not None
                ]
            ),
        )

    @property
    def body(self) -> StreamingBody:
        '''
        The cached StreamingBody, which may have already been consumed. Prefer bytes(self).
        '''
        return self.obj['Body']

    @property
    def last_modified(self) -> datetime:
        '''
        The datetime that the object was last modified, corresponding to the point in time that SES
        received the message and inserted it into S3.
        '''
        return self.obj['LastModified']


def insert_message(ses_msg: SESMessage, metadata: dict[str, Any]) -> GmailMessage:
    '''
    Insert the given SES message into Gmail with the given metadata and modified content.
    '''

    # This block is not a critical section, and must handle the possibility that it runs twice for
    # the same message concurrently. Unfortunately, due to gmail's preference to replace poorly
    # formed Message-ID headers with random values, we cannot trivially recover messages that get
    # inserted, then lost (either due to a runtime environment failure or due to concurrent writes).
    # As a result, the deduplication logic is imperfect, and fixing it would likely require the use
    # of dynamodb with a strongly consistent conditional write.
    with google_session() as sess:
        # TODO: handle message threading
        encoder = MultipartRelatedEncoder(
            fields=[
                (None, (None, json.dumps(metadata), 'application/json')),
                (None, (None, ses_msg.buffer, 'message/rfc822')),
            ]
        )
        start_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        ses_msg.set_tags(dict(Forwarded=f'start:{start_at}'))
        res = sess.post(
            'https://gmail.googleapis.com/upload/gmail/v1/users/me/messages',
            headers={'content-type': encoder.content_type},
            data=encoder,
            params=dict(internalDateSource='dateHeader'),
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()

        msg = GmailMessage(data['id'], data['threadId'])
        print(f'Created message {msg.message_id} in thread {msg.thread_id}')

        forward_success = msg.rfc822_message_id == ses_msg.rfc822_message_id
        ses_msg.set_tags(
            dict(Forwarded=str(forward_success).lower(), GmailMessageID=msg.message_id)
        )

        # Message contained a bad rfc822 message ID, and it was overwritten.
        if forward_success:
            print('Marked S3 object for deletion')
        else:
            # TODO: take some extra action to recover here.
            print('Created message had mismatched message ID')

        return msg


ForwardingDisposition = Enum('ForwardingDisposition', 'PROCEED RECOVER WAIT SKIP')


def forward_email(
    ses_msg: SESMessage, disposition: ForwardingDisposition = ForwardingDisposition.PROCEED
) -> bool:
    '''
    Forward the email with the given SES message ID, corresponding to the S3 object key. Optionally
    verify that the email has not already been forwarded when `proactive_duplicate_check=True`.
    '''

    print(f'Received message {ses_msg.key}')

    pmsg = ses_msg.email_message()

    print(f'Processing message as {ses_msg.rfc822_message_id}')

    if disposition == ForwardingDisposition.RECOVER:
        messages_matched = deduplicate_email(
            ses_msg.rfc822_message_id, ses_id=pmsg['x-ses-receipt']
        )
        if messages_matched:
            print(f'Proactive duplicate check matched {messages_matched} messages')
            return False
        if ses_msg.get_tags().get('GmailMessageID'):
            print(
                'Proactive duplicate check detected existing message, but did not recover it from '
                'its Message-ID header'
            )
            return False

        # At least two ways to get here: either we started inserting the message but got
        # interrupted, or we succeeded in inserting the message but the success got overwritten by
        # a duplicate forwarding call.

    mark_as_spam = (
        pmsg.get('x-ses-spam-verdict') != 'PASS' or pmsg.get('x-ses-virus-verdict') != 'PASS'
    )
    msg_label_ids = label_ids + ['SPAM' if mark_as_spam else 'INBOX']
    message_metadata = dict(labelIds=msg_label_ids)

    obj_date = ses_msg.last_modified

    date_header = pmsg.get('date')
    parsed_date = date_header and parsedate_to_datetime(date_header)
    update_timestamp = (
        not parsed_date
        or abs(parsed_date.timestamp() - obj_date.timestamp()) > ACCEPTABLE_DELIVERY_DELAY
    )
    if parsed_date and update_timestamp:
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
        ses_msg.buffer = RawBytesGenerator.convert_bytes(
            pmsg, linesep=infer_linesep(bytes(ses_msg))
        )

    if disposition == ForwardingDisposition.PROCEED:
        # Not important if this happens multiple times, because it'll be consistent. These tags are
        # useful for processing by other systems, especially when coupled with EVENTS_TOPIC_ARN.
        sender = getaddresses([pmsg.get('from') or ''])
        ses_msg.set_tags(
            dict(
                Sender=(sender and sender[0][1]) or None,
                Subject=pmsg.get('subject'),
                RFC822MessageID=ses_msg.rfc822_message_id,
            ),
        )

    # Note that this may occur multiple times, and the consumer is responsible for deduplicating
    # events.
    if not mark_as_spam and disposition == ForwardingDisposition.PROCEED and EVENTS_TOPIC_ARN:
        # publish to sns
        sns_client.publish(
            TopicArn=EVENTS_TOPIC_ARN,
            Message=json.dumps(dict(type="receive", bucket=ses_msg.bucket, key=ses_msg.key)),
        )

    _gmail_message = insert_message(ses_msg=ses_msg, metadata=message_metadata)

    matched_copies = deduplicate_email(ses_msg.rfc822_message_id)

    if not matched_copies:
        print(f'No messages found for rfc822 message id `{ses_msg.rfc822_message_id}`')
    elif matched_copies > 3:
        # TODO: this should really notify
        print(
            'Tried to deduplicate more than three messages for rfc822 '
            f'message id `{ses_msg.rfc822_message_id}`!'
        )
    return True


def get_forwarding_disposition(event: dict[str, Any], ses_msg: SESMessage) -> ForwardingDisposition:
    '''
    Messages may be forwarded, unforwarded, or in the process of being forwarded. If the message is
    actively being forwarded, wait for the delivery timeout. Note that eventual consistency means we
    cannot guarantee that this check is accurate.
    '''
    if event.get('ignoreTags', False):
        return ForwardingDisposition.PROCEED
    forwarded = ses_msg.get_tags().get('Forwarded', '')
    if forwarded in {'true', 'false'}:
        # The 'false' value indicates a previous fatal error, and that a retry should not be
        # attempted.
        return ForwardingDisposition.SKIP
    if not forwarded.startswith('start:'):
        return ForwardingDisposition.PROCEED
    retry_after = (
        datetime.fromisoformat(forwarded[len('start:') :].replace('Z', '+00:00'))
        + GMAIL_DELIVERY_TIMEOUT
    )
    return (
        ForwardingDisposition.RECOVER
        if retry_after < datetime.now(timezone.utc)
        else ForwardingDisposition.WAIT
    )


def get_event_messages(event: dict[str, Any]) -> Iterable[SESMessage]:
    '''
    Iterate over the SES messages for the given event.
    '''
    records = event.get('Records')
    if not records:
        print('Processing all remaining records')
        # Assume we need to comb through the S3 bucket for recovery purposes.
        for page in s3_client.get_paginator('list_objects_v2').paginate(
            Bucket=s3_bucket, Prefix=s3_prefix, ExpectedBucketOwner=account_id
        ):
            for obj in page['Contents']:
                yield SESMessage(bucket=s3_bucket, key=obj['Key'])
    else:
        print(f'Processing {len(records)} records')
        yield from (
            SESMessage(bucket=s3_bucket, key=record['ses']['mail']['messageId'])
            for record in records
        )


def lambda_handler(event: dict[str, Any], context: Any) -> None:  # pylint: disable=unused-argument
    '''
    Handle the Lambda invocation, either via SES -> SNS, or via manual invocation to handle
    operational outages/authentication failure gaps.
    '''

    dispositions = set()
    for ses_msg in get_event_messages(event):
        disposition = get_forwarding_disposition(event, ses_msg)
        dispositions.add(disposition)
        if disposition in (ForwardingDisposition.PROCEED, ForwardingDisposition.RECOVER):
            # TODO: don't re-forward deleted emails. This will only happen if we are
            # unsuccessful in recording the result from Gmail's API as a Forwarded=true.
            forward_email(ses_msg, disposition)

    # TODO: improve retry strategy
    if 'Records' in event and ForwardingDisposition.WAIT in dispositions:
        raise Exception('retrying SES request due to WAIT disposition')
