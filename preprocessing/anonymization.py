"""Identity replacement and station-specific timestamp translation.

Scientific values are copied unchanged. Original identifiers and time offsets
stay in private mappings outside the public file tree.
"""
from datetime import datetime, timedelta, timezone
import numpy as np


def parse_time(value):
    result=value if isinstance(value,datetime) else datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is not None:result=result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def station_anchors(count,seed=20261004):
    first=datetime(2001,1,1)
    span=(datetime(2011,1,1)-first).days
    if not 0<count<=span:raise ValueError('Invalid number of distinct station anchors')
    # A separate stream keeps date assignment independent of station sampling.
    rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
    return [first+timedelta(days=int(day)) for day in rng.choice(span,size=count,replace=False)]


def site_anchor_mapping(site_ids=None,seed=20261004):
    """Stable anchors for the fixed public namespace, independent of input order.

    Requesting a subset of IDs retains their anchors from the full namespace.
    """
    canonical=tuple(f'SITE_{i:03d}' for i in range(1,13))
    requested=list(canonical if site_ids is None else site_ids)
    if len(set(requested))!=len(requested) or not set(requested).issubset(canonical):
        raise ValueError('Expected unique anonymous site IDs from SITE_001 to SITE_012')
    full=dict(zip(canonical,station_anchors(len(canonical),seed)))
    return {site:full[site] for site in sorted(requested)}


def shifted_time(value,source_first,anchor):
    return parse_time(anchor)+(parse_time(value)-parse_time(source_first))


def iso_time(value):
    return parse_time(value).isoformat(timespec='microseconds')+'Z'
