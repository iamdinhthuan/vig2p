# vig2p

Vietnamese G2P helpers with the Kokoro-compatible phoneme adapter used by the
Vietnamese Kokoro fine-tuning workflow.

## Install

```bash
pip install vig2p
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
vig2p "tường nhà khách. thường nhà khách."
python -m vig2p "trước chước số xố giải dải"
```

Read one utterance per line:

```bash
vig2p --file metadata_texts.txt
```

JSON output:

```bash
vig2p --json "tường nhà khách"
```

## Python

```python
from vig2p import VietnameseG2P, phonemize_text

print(phonemize_text("tường nhà khách"))

g2p = VietnameseG2P()
print(g2p("trước chước số xố giải dải"))
```

The default engine is intentionally open for the next native Vietnamese G2P
implementation. Adapter helpers can be tested with an injected engine that
implements `run(text: str)`.

## Adapter Policy

- `tʃ -> ʧ`
- `t̪ -> t`
- source words starting `th` rewrite first raw `t -> θ`, including unaccented Vietnamese words like `theo`
- source words starting `tr` rewrite first raw `ʧ -> ʈʂ`
- source words starting `s` rewrite first raw `s -> ʂ`, including unaccented Vietnamese words like `sinh`; obvious non-Vietnamese `s` clusters such as `st/sp/sk/...` are left alone
- source words starting `gi`/`gì` variants rewrite first raw `z -> ʝ`
- `e- -> æ`
- `1/7 -> →`, `2 -> ↘`, `3/ɜ -> ↗`, `4 -> ↓`, `5 -> ʔ↗`, `6 -> ʔ↓`
- `ɗ -> d`, `ʐ -> ʒ`, `đ -> d`
- curly apostrophes behave like apostrophes
- `– -> —`
- unsupported `*`, `/`, `&` are stripped or spaced before training phonemes


## Test

```bash
python -m unittest discover -s tests -v
```

## Publish

Build and check the package:

```bash
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then upload to PyPI:

```bash
python -m twine upload dist/*
```
