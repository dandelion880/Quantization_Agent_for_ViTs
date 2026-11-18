"""Fixed prefix intervals and a variable-length tail; inference uses only fitted grids."""
import torch
from ..schemas import require
from .grids import affine_params, affine_qdq


def spans(length, ends):
    require(ends == sorted(set(ends)) and 0 < ends[0] and ends[-1] < length,
            'static token partition requires ordered nonempty groups and a nonempty tail')
    return zip([0, *ends], [*ends, length])


def fit(x, spec, partition, percentiles=(1.,)):
    value = x.detach().float().movedim(partition['axis'], 1)
    grids = []
    for start, stop in spans(value.shape[1], partition['prefix_ends']):
        parts = []
        group = value[:, start:stop].reshape(-1, value[0, 0].numel())
        for columns in group.split(64, dim=1):
            best = None
            for p in percentiles:
                if p == 1:
                    lo, hi = columns.amin(0), columns.amax(0)
                else:
                    lo, hi = torch.quantile(columns, columns.new_tensor([1-p, p]), dim=0)
                params = affine_params(lo.clamp_max(0), hi.clamp_min(0), spec['bits'])
                error = (affine_qdq(columns, spec['bits'], params) - columns).square().mean(0)
                if best is None:
                    best, selected = error, params
                else:
                    improve = error < best
                    selected = {k: torch.where(improve, params[k], selected[k]) for k in params}
                    best = torch.minimum(error, best)
            parts.append(selected)
        grids.append({k: torch.cat([row[k] for row in parts]).reshape(value.shape[2:]) for k in parts[0]})
    return {k: torch.stack([row[k] for row in grids]) for k in grids[0]}


def apply(x, spec, partition, params):
    axis = partition['axis']
    value = x.movedim(axis, 1)
    pieces = [affine_qdq(value[:, start:stop], spec['bits'], {k: v[i] for k, v in params.items()})
              for i, (start, stop) in enumerate(spans(value.shape[1], partition['prefix_ends']))]
    return torch.cat(pieces, dim=1).movedim(1, axis).to(x.dtype)


def validate(params, spec, partition, shape):
    axis, ends = partition['axis'], partition['prefix_ends']
    require(0 < axis < len(shape), 'token axis outside observed shape')
    list(spans(shape[axis], ends))
    require(spec['axis'] == max(i for i in range(1, len(shape)) if i != axis),
            'partitioned affine expects the channel axis to be the last non-token feature dimension')
    expected = (len(ends) + 1, *(n for i,n in enumerate(shape) if i not in (0,axis)))
    require(set(params) == {'scale','zero'} and all(tuple(v.shape) == expected and torch.isfinite(v).all()
            for v in params.values()), 'frozen token partition shape/values mismatch')
    zero = params['zero']
    require(bool((params['scale'] > 0).all() and ((zero == zero.round()) & (zero >= 0)
            & (zero < 2 ** spec['bits'])).all()), 'partition scale/zero outside affine grid')
