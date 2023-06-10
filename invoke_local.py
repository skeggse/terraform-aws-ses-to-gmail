#!/usr/bin/env python3

import sys
from os import environ, execve

import boto3

lambda_client = boto3.client('lambda', region_name='us-west-2')


def invoke_exec():
    assert (
        len(sys.argv) > 1
        and (sys.argv[1] == 'main.py' or sys.argv[1].startswith('test_'))
        and sys.argv[1].endswith('.py')
    )
    if sys.argv[1] == 'main.py':
        args = sys.argv[1:]
    else:
        args = ['-m', 'unittest', *sys.argv[1:]]
    env_vars = (
        lambda_client.get_function(FunctionName='forward-emails')['Configuration'][
            'Environment'
        ].get('Variables')
        or {}
    )
    execve(sys.executable, ['python3', *args], env={**environ, **env_vars})


if __name__ == '__main__':
    invoke_exec()
