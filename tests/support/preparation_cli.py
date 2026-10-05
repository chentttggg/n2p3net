"""Build explicit option/value argv pairs for preparation entrypoint tests."""


def cli_options(**options):
    return [
        token
        for name, value in options.items()
        for token in (f"--{name.replace('_', '-')}", str(value))
    ]


def set_preparation_cli(monkeypatch, script, *, sources=(), **options):
    source_args = [token for source in sources for token in ("--source", str(source))]
    monkeypatch.setattr("sys.argv", [script, *source_args, *cli_options(**options)])
