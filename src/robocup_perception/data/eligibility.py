"""Metadata-only grouping and split checks; never infers independent exposure."""
from collections import defaultdict


def components(records, verified_pairs=()):
    """Connect sessions, exact hashes and explicitly verified duplicate pairs.

    Records require id. Optional session and pixel_sha256 connect identities.
    Perceptual similarity alone must not be passed as a verified pair.
    """
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate record IDs')
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        if a not in parent or b not in parent:
            raise ValueError('Relationship references an unknown identity')
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for field in ('session', 'pixel_sha256'):
        seen = {}
        for row in records:
            value = row.get(field)
            if value:
                if value in seen:
                    union(row['id'], seen[value])
                seen[value] = row['id']
    for a, b in verified_pairs:
        union(a, b)
    return {key: find(key) for key in ids}


def audit(records, verified_pairs=()):
    group = components(records, verified_pairs)
    splits, exposed = defaultdict(set), defaultdict(set)
    for row in records:
        component = group[row['id']]
        if row.get('split'):
            splits[component].add(row['split'])
        exposed[component].update(row.get('exposures', []))
    return {
        'components': group,
        'cross_split_components': {k: sorted(v) for k, v in splits.items() if len(v) > 1},
        'component_exposures': {k: sorted(v) for k, v in exposed.items()},
        'independence_warning': 'Absence of recorded exposure is not proof of an unused capture.'
    }
