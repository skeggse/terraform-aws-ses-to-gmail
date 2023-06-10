from os import environ
import re
import unittest

import boto3

from main import RawBytesGenerator, SESMessage, infer_linesep

s3 = boto3.client('s3', region_name='us-west-2')

trailing_whitespace = re.compile(rb'[ \t]+(\r)$', re.MULTILINE)

TEST_CASES_BUCKET = environ['TEST_CASES_BUCKET']


def trimmed(value: bytes) -> bytes:
    return trailing_whitespace.sub(rb'\1', value)


class TestEncoding(unittest.TestCase):
    def test_quoted_printable_ascii(self):
        msg = SESMessage(bucket=TEST_CASES_BUCKET, key='ascii_quoted_printable_test_case')
        parsed_msg = msg.email_message()
        converted = RawBytesGenerator.convert_bytes(parsed_msg, infer_linesep(bytes(msg)))
        self.assertEqual(trimmed(bytes(msg)), trimmed(converted))

    def test_base64_utf8(self):
        msg = SESMessage(bucket=TEST_CASES_BUCKET, key='rejected_message_id_test_case')
        parsed_msg = msg.email_message()
        converted = RawBytesGenerator.convert_bytes(parsed_msg, infer_linesep(bytes(msg)))
        self.assertEqual(trimmed(bytes(msg)), trimmed(converted))

    def test_quoted_printable_utf8(self):
        msg = SESMessage(bucket=TEST_CASES_BUCKET, key='utf8_quoted_printable_test_case')
        parsed_msg = msg.email_message()
        converted = RawBytesGenerator.convert_bytes(parsed_msg, infer_linesep(bytes(msg)))
        self.assertEqual(trimmed(bytes(msg)), trimmed(converted))

    def test_utf8(self):
        msg = SESMessage(bucket=TEST_CASES_BUCKET, key='utf8_test_case')
        parsed_msg = msg.email_message()
        converted = RawBytesGenerator.convert_bytes(parsed_msg, infer_linesep(bytes(msg)))
        self.assertEqual(trimmed(bytes(msg)), trimmed(converted))
