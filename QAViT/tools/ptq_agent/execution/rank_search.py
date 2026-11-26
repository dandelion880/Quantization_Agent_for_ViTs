"""AIQViT Eq. 11–14, bounded first-order bilevel calibration on disjoint image IDs."""

import torch
from ..schemas import require


def search(block, members, rows, config, gen):
    by_id = {row[0]: row for row in rows}
    train, val = (config[key] for key in ("weight_ids", "architecture_ids"))
    require(
        train
        and val
        and set(train).isdisjoint(val)
        and (set(train) | set(val)) <= by_id.keys(),
        "rank search needs disjoint declared splits of the calibration view",
    )
    p = config["drop_path"]
    require(0 <= p < 1 and config["steps"] > 0, "invalid search schedule")
    choices, alphas, branches = {}, {}, {}
    for sid, q in members.items():
        ranks = config["candidates"][sid]
        w = q.reference_weight
        require(
            ranks
            and len(set(ranks)) == len(ranks)
            and all(0 < r <= min(w.shape) for r in ranks),
            "invalid rank proposal",
        )
        choices[sid] = ranks
        alphas[sid] = torch.nn.Parameter(w.new_zeros(len(ranks)))
        branches[sid] = [
            (
                torch.nn.Parameter(
                    torch.randn((r, w.size(1)), device=w.device, generator=gen)
                    * config["init_std"]
                ),
                torch.nn.Parameter(w.new_zeros(w.size(0), r)),
            )
            for r in ranks
        ]
    parameters = [t for group in branches.values() for pair in group for t in pair]
    wopt = torch.optim.Adam(parameters, lr=config["weights_lr"])
    aopt = torch.optim.Adam(alphas.values(), lr=config["architecture_lr"])

    def objective(ids, training):
        for sid, q in members.items():
            coeff = alphas[sid].softmax(0)
            # A scalar Bernoulli per branch is shared by the single-image forward.
            mask = (
                (torch.rand(coeff.shape, generator=gen, device=coeff.device) >= p).to(
                    coeff
                )
                / (1 - p)
                if training
                else torch.ones_like(coeff)
            )
            q.decoded_weight = q.reference_weight + sum(
                a * m * (B @ A) for a, m, (A, B) in zip(coeff, mask, branches[sid])
            )
        values = []
        for sid in ids:
            _, _, x, y, args, kwargs = by_id[sid]
            values.append((block(x, *args, **kwargs) - y).norm())
        return torch.stack(values).mean()

    losses = []
    for _ in range(config["steps"]):
        wopt.zero_grad()
        loss = objective(train, True)
        loss.backward()
        wopt.step()
        aopt.zero_grad()
        v = objective(val, False)
        v.backward()
        aopt.step()
        losses.append([float(loss.detach()), float(v.detach())])
    winner = {k: int(a.detach().argmax()) for k, a in alphas.items()}
    selected = {
        k: [torch.nn.Parameter(t.detach()) for t in branches[k][i]]
        for k, i in winner.items()
    }
    return selected, {
        "chosen_ranks": {k: choices[k][i] for k, i in winner.items()},
        "alphas": {k: v.softmax(0).detach().cpu().tolist() for k, v in alphas.items()},
        "weight_ids": train,
        "architecture_ids": val,
        "losses": losses,
        "approximation": "alternating first-order architecture gradient",
    }


def bounds(variable):
    low, width = variable.sigmoid().unbind()
    return low, low + (1 - low) * width


def focus(x, variable, bits):
    lo, hi = bounds(variable)
    n = (1 << bits) - 1
    v = ((x - lo) / (hi - lo)).clamp(0, 1) * n
    v = v + (v.round() - v).detach()
    return v / n * (hi - lo) + lo
