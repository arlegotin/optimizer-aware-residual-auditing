# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic byte articles and target blocks; no data access on import.

Encode articles once during preprocessing, then reuse their token tuples for
collation. Public manifests contain provenance and counts, never article text.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import re


_HEADING = re.compile(r'^= ([^=\s](?:.*?\S)?) =$')
_CONTEXT = 256
_BOS, _EOS, _PAD = 256, 257, 258


@dataclass(frozen=True)
class Article:
    source_split: str
    source_article_index: int
    sha256: str
    text: str


@dataclass(frozen=True)
class TargetBlock:
    article_index: int
    target_start: int
    target_end: int
    input_start: int


def normalize_articles(rows: Iterable[str], source_split: str) -> list[Article]:
    """Recognize headings per source row, preserving every normalized character."""
    articles = []
    lines = []

    def emit() -> None:
        text = ''.join(lines)
        if text:
            articles.append(Article(source_split, len(articles),
                                    hashlib.sha256(text.encode('utf8')).hexdigest(), text))
        lines.clear()

    for row in rows:
        if not isinstance(row, str):
            raise TypeError('each source row must be a string')
        normalized = row.replace('\r\n', '\n').replace('\r', '\n')
        for line in normalized.splitlines(keepends=True):
            if _HEADING.fullmatch(line.strip()):
                emit()
            lines.append(line)
    emit()
    return articles


def encode_articles(articles: Sequence[Article]) -> list[tuple[int, ...]]:
    """Pay UTF8 tokenization once, with one BOS and one EOS per article."""
    return [(_BOS, *article.text.encode('utf8'), _EOS) for article in articles]


def split_training_articles(articles: Sequence[Article]) -> tuple[list[Article], list[Article]]:
    fitting, development = [], []
    for article in articles:
        (development if int(article.sha256, 16) % 10 == 0 else fitting).append(article)
    return fitting, development


def _provenance(article: Article) -> dict:
    return {'split': article.source_split, 'source_article_index': article.source_article_index,
            'sha256': article.sha256}


def exclude_training_duplicates(evaluation: Sequence[Article], training: Sequence[Article]) -> tuple[list[Article], list[dict]]:
    """Exclude hashes in all training; retain multiplicity among evaluation rows."""
    training_hashes = {article.sha256 for article in training}
    kept, exclusions = [], []
    for article in evaluation:
        if article.sha256 in training_hashes:
            exclusions.append(_provenance(article))
        else:
            kept.append(article)
    return kept, exclusions


def training_chunks(articles: Sequence[Article]) -> list[TargetBlock]:
    blocks = []
    for index, article in enumerate(articles):
        end = len(article.text.encode('utf8'))+1
        for start in range(1, end+1, _CONTEXT):
            blocks.append(TargetBlock(index, start, min(start+_CONTEXT-1, end), start-1))
    return blocks


def evaluation_windows(articles: Sequence[Article]) -> list[TargetBlock]:
    blocks = []
    for index, article in enumerate(articles):
        end = len(article.text.encode('utf8'))+1
        for start in range(1, end+1, 128):
            stop = min(start+127, end)
            blocks.append(TargetBlock(index, start, stop, max(0, stop-_CONTEXT)))
    return blocks


def _validate_block(block: TargetBlock, article_count: int, token_count: int | None = None) -> None:
    if not isinstance(block, TargetBlock):
        raise TypeError('block must be a TargetBlock')
    fields = (block.article_index, block.target_start, block.target_end, block.input_start)
    if any(type(value) is not int for value in fields):
        raise TypeError('block bounds must be integers')
    if not 0 <= block.article_index < article_count:
        raise ValueError('block article index is out of range')
    if not 0 <= block.input_start < block.target_start <= block.target_end:
        raise ValueError('block target/input bounds are invalid')
    if not 1 <= block.target_end-block.input_start <= _CONTEXT:
        raise ValueError('block input length must be between 1 and 256')
    if token_count is not None and block.target_end >= token_count:
        raise ValueError('block target end exceeds article EOS')


def choose_monitor(articles: Sequence[Article], windows: Sequence[TargetBlock], maximum: int = 64) -> list[TargetBlock]:
    if type(maximum) is not int:
        raise TypeError('monitor maximum must be an integer')
    if not 1 <= maximum <= 64:
        raise ValueError('monitor maximum must be between 1 and 64')
    # Token lengths are computed once per article, not once per window.
    lengths = [len(article.text.encode('utf8'))+2 for article in articles]
    for window in windows:
        _validate_block(window, len(articles))
        _validate_block(window, len(articles), lengths[window.article_index])

    def key(window: TargetBlock) -> tuple:
        article = articles[window.article_index]
        digest = hashlib.sha256((article.sha256+':'+str(window.target_start)).encode('ascii')).digest()
        return digest, article.sha256, window.target_start, article.source_article_index

    selected = sorted(windows, key=key)[:maximum]
    return sorted(selected, key=lambda window: (window.article_index, window.target_start))


def collate_blocks(tokenized_articles: Sequence[Sequence[int]], blocks: Sequence[TargetBlock], device='cpu') -> dict:
    """Right-pad actual rows; target masks mark scored logits, not key validity."""
    if not blocks:
        raise ValueError('cannot collate an empty batch')
    inputs, targets, masks = [], [], []
    valid_targets = 0
    for block in blocks:
        _validate_block(block, len(tokenized_articles))
        tokens = tokenized_articles[block.article_index]
        _validate_block(block, len(tokenized_articles), len(tokens))
        if (len(tokens) < 2 or type(tokens[0]) is not int or type(tokens[-1]) is not int
                or tokens[0] != _BOS or tokens[-1] != _EOS):
            raise ValueError('tokenized article must have BOS and EOS boundaries')
        # Inspect only this block and its final target, never the whole article.
        section = tokens[block.input_start:block.target_end+1]
        for offset, token in enumerate(section, start=block.input_start):
            if type(token) is not int:
                raise TypeError('article tokens must be integers')
            expected_boundary = _BOS if offset == 0 else _EOS if offset == len(tokens)-1 else None
            if (expected_boundary is not None and token != expected_boundary) or (
                    expected_boundary is None and not 0 <= token < 256):
                raise ValueError('article token is not a byte or the correct boundary')
        input_length = block.target_end-block.input_start
        row_inputs = list(section[:-1])+[_PAD]*(_CONTEXT-input_length)
        row_targets = [-100]*_CONTEXT
        row_mask = [False]*_CONTEXT
        for target in range(block.target_start, block.target_end+1):
            position = target-block.input_start-1
            row_targets[position] = tokens[target]
            row_mask[position] = True
        inputs.append(row_inputs)
        targets.append(row_targets)
        masks.append(row_mask)
        valid_targets += block.target_end-block.target_start+1

    import torch

    return {'input_ids': torch.tensor(inputs, dtype=torch.int64, device=device),
            'target_ids': torch.tensor(targets, dtype=torch.int64, device=device),
            'target_mask_cpu': torch.tensor(masks, dtype=torch.bool, device='cpu'),
            'valid_targets': valid_targets, 'physical_rows': len(blocks)*_CONTEXT}


def article_manifest(articles: Sequence[Article]) -> list[dict]:
    result = []
    for article in articles:
        size = len(article.text.encode('utf8'))
        result.append({**_provenance(article), 'utf8_bytes': size, 'targets': size+1})
    return result


def block_manifest(articles: Sequence[Article], blocks: Sequence[TargetBlock]) -> list[dict]:
    lengths = [len(article.text.encode('utf8'))+2 for article in articles]
    result = []
    for block in blocks:
        _validate_block(block, len(articles))
        _validate_block(block, len(articles), lengths[block.article_index])
        result.append({**_provenance(articles[block.article_index]),
                       'target_start': block.target_start, 'target_end': block.target_end,
                       'input_start': block.input_start,
                       'input_length': block.target_end-block.input_start,
                       'scored_targets': block.target_end-block.target_start+1})
    return result
