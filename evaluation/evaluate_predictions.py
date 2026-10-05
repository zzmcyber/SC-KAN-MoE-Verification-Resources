#!/usr/bin/env python3
"""Evaluate finite, uniquely identified observation-prediction pairs without clipping."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np


def is_constant(values):
    tolerance=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(values))))
    return bool(np.ptp(values)<=tolerance)


def metrics(observed,predicted):
    observed=np.asarray(observed,dtype=float);predicted=np.asarray(predicted,dtype=float)
    if observed.ndim!=1 or predicted.ndim!=1 or observed.shape!=predicted.shape or observed.size==0:
        raise ValueError('Expected nonempty one-dimensional arrays of equal length')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Observation and prediction values must be finite')
    error=predicted-observed
    result={'n':int(observed.size),'RMSE':float(np.sqrt(np.mean(error**2))),
            'R2':None,'Pearson_r':None,'KGE':None,'Bias':float(error.mean())}
    constant_observed=is_constant(observed);constant_predicted=is_constant(predicted)
    if observed.size>=2 and not constant_observed:
        result['R2']=float(1-np.sum(error**2)/np.sum((observed-observed.mean())**2))
        if not constant_predicted:
            association=float(np.corrcoef(observed,predicted)[0,1])
            result['Pearson_r']=association
            mean_tolerance=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(observed))))
            if abs(observed.mean())>mean_tolerance:
                alpha=float(predicted.std(ddof=1)/observed.std(ddof=1))
                beta=float(predicted.mean()/observed.mean())
                result['KGE']=1-math.sqrt((association-1)**2+(alpha-1)**2+(beta-1)**2)
    return result


def validate(rows):
    if not rows:raise ValueError('No observation-prediction pairs')
    required={'sample_id','site_id','observed_sm','predicted_sm'}
    ids=[]
    for row in rows:
        if not required.issubset(row) or any(row[k] is None or str(row[k]).strip()=='' for k in required):
            raise ValueError('Missing sample ID, site ID, observation, or prediction')
        ids.append(row['sample_id'])
        try:values=[float(row['observed_sm']),float(row['predicted_sm'])]
        except (ValueError,TypeError):raise ValueError('Observation and prediction values must be numeric') from None
        if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite observation or prediction')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample IDs')
    for field in ('split','model_type','training_scope'):
        values={str(row.get(field,'')) for row in rows}
        if len(values)>1:raise ValueError(f'Mixed {field}; select one population before evaluation')


def evaluate(rows,by_site=False):
    validate(rows)
    def compute(group):return metrics([float(r['observed_sm']) for r in group],[float(r['predicted_sm']) for r in group])
    if not by_site:return {'aggregation':'pooled',**compute(rows)}
    groups=defaultdict(list)
    for row in rows:groups[row['site_id']].append(row)
    return {'aggregation':'per_site','sites':{site:compute(group) for site,group in sorted(groups.items())}}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--by-site',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:result=evaluate(read_csv(args.csv),args.by_site)
    except (OSError,ValueError) as error:parser.error(str(error))
    output=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':main()
