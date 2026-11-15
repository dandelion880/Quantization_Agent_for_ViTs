#!/usr/bin/env python3
"""Read-only CPU comparison of serialized model payloads; never runs inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    files = [args.source, args.candidate]
    before = [sha(f) for f in files]
    values = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    changes = []
    counts = {'tensor_pairs': 0, 'equal_tensor_pairs': 0, 'non_tensor_leaf_pairs': 0}

    def compare(a, b, path=''):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            counts['tensor_pairs'] += 1
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            counts['equal_tensor_pairs'] += int(equal)
            if not equal:
                changes.append({'path': path, 'kind': 'tensor', 'shape_before': list(a.shape),
                    'shape_after': list(b.shape), 'dtype_before': str(a.dtype), 'dtype_after': str(b.dtype),
                    'different_elements': int((a != b).sum()) if a.shape == b.shape else None,
                    'max_abs_difference': float((a.double()-b.double()).abs().max())
                        if a.shape == b.shape and a.numel() else None})
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    changes.append({'path': path + '/' + str(key), 'kind': 'missing_key'})
                else:
                    compare(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                changes.append({'path': path, 'kind': 'sequence_length'})
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, path + '/' + str(index))
        else:
            counts['non_tensor_leaf_pairs'] += 1
            if type(a) != type(b) or a != b:
                changes.append({'path': path, 'kind': 'scalar', 'before': a, 'after': b})

    compare(*values)
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Read-only CPU serialized payload comparison; no inference or scientific experiment. Does not replace reload or full metrics.',
        'files': [str(f) for f in files], 'sha256_before': before,
        'sha256_after': [sha(f) for f in files], 'counts': counts, 'changes': changes,
        'all_changes_within_target_Q': all('/blocks.0.attn::attention_query/' in x['path'] for x in changes)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
