"""Check Japan Cloud credentials without invoking a model or uploading business traces."""
import os
from pathlib import Path


def main():
    path = Path('.env.langfuse')
    allowed = {'RETAILFLOW_LANGFUSE_ENABLED', 'LANGFUSE_BASE_URL', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'}
    if path.exists():
        # Literal assignments only: never execute this file as shell code.
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if separator and key.strip() in allowed:
                os.environ[key.strip()] = value.strip().strip('\"\'')
    for key in ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'):
        if not os.environ.get(key):
            print(f'Missing {key}; fill it in .env.langfuse locally.')
            return 1
    try:
        from retailflow.observability import _connection
        client, _ = _connection()
        if not client.auth_check():
            print('Authentication failed; check region and project keys locally.')
            return 1
    except Exception as exc:
        print(f'Connection check failed ({type(exc).__name__}); no key values printed.')
        return 1
    print('Langfuse project authentication succeeded. No model or business task was run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
