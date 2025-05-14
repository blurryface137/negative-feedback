import torch, tqdm
from ntrsasrec import NTRSASRec
from ir_measures import ScoredDoc, Qrel
import ir_measures


@torch.no_grad()
def evaluate(model: NTRSASRec, loader, metrics, k, filter_rated, device):
    model.eval()
    scored, qrels = [], []
    uid = 0
    for items, actions, rated, target in tqdm.tqdm(loader):
        items, actions, target = items.to(device), actions.to(device), target.to(device)
        indices, scores = model.get_predictions(items, actions, k, rated if filter_rated else None)

        for idxs, scs, tgt in zip(indices, scores, target):
            for itm, s in zip(idxs.tolist(), scs.tolist()):
                scored.append(ScoredDoc(str(uid), str(itm), float(s)))
            qrels.append(Qrel(str(uid), str(tgt.item()), 1))
            uid += 1
    return ir_measures.calc_aggregate(metrics, qrels, scored)
