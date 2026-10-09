"""Scientific tokenizer identity survives credential redaction."""
from fedcore.experiments.runner import _redact


def test_tokenizer_metadata_kept_and_nested_auth_fields_redacted():
    record = {'tokenizer': {'name': 'utf8-byte-v1', 'access_token': 'private'},
              'token_count': 42, 'api_key': 'private', 'github_token': 'private'}
    assert _redact(record) == {
        'tokenizer': {'name': 'utf8-byte-v1', 'access_token': '[redacted]'},
        'token_count': 42, 'api_key': '[redacted]', 'github_token': '[redacted]'}
    assert record['api_key'] == 'private'
