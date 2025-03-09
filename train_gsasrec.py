from argparse import ArgumentParser
import os

import torch
from torchinfo import summary
from tqdm import tqdm

from utils import load_config, build_model, get_device
from dataset_utils import get_train_dataloader, get_val_dataloader, get_num_items
from eval_utils import evaluate

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_retailrocket.py')
parser.add_argument('--use_predefined_neg', action='store_true', default=False,
                    help="при установке флага будем использовать заранее определённые негативы.")

if any('jupyter' in arg for arg in sys.argv) or any('ipykernel' in arg for arg in sys.argv):
    args, unknown = parser.parse_known_args([])
else:
    args, unknown = parser.parse_known_args()

config = load_config(args.config)
num_items = get_num_items(config.dataset_name)
device = get_device()
model = build_model(config).to(device)

train_dataloader = get_train_dataloader(
    config.dataset_name,
    batch_size=config.train_batch_size,
    max_length=config.sequence_length
)
val_dataloader = get_val_dataloader(
    config.dataset_name,
    batch_size=config.eval_batch_size,
    max_length=config.sequence_length
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

def not_to_recommend_loss(logits, labels, mask):
    """
    logits shape: [B, max_len]
    labels shape: [B, max_len] (0 или 1)
    mask   shape: [B, max_len], 1=считаем в лоссе, 0=пропускаем (padding)
    sum( y*log(sigmoid(logit)) + (1-y)*log(1 - sigmoid(logit)) )
    но в варианте "p = softmax(...)?"
    В gSASRec p обычно получается через скалярное произведение с эмбеддингами,
    но можно и sigmoid. Можно делать logits -> sigma,
    затем -log(sigma) или -log(1-sigma).

    реализация BCE:
    BCE(pos) = -log(sigmoid(logit)),
    BCE(neg) = -log(1 - sigmoid(logit)).
    """
    # считаем p = sigmoid(logits)
    # logits[B, max_len]
    preds = torch.sigmoid(logits)
    # labels: 1 => -log(preds), 0 => -log(1 - preds)
    eps = 1e-8
    loss_pos = -labels * torch.log(preds + eps)
    loss_neg = -(1 - labels) * torch.log(1 - preds + eps)
    loss_elm = loss_pos + loss_neg  # [B, max_len]
    loss_elm = loss_elm * mask      # убираем паддинги
    loss_mean = loss_elm.sum() / (mask.sum() + eps)
    return loss_mean

for epoch in range(config.max_epochs):
    model.train()
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0.0

    for _ in pbar:
        step += 1
        try:
            pos_seq, neg_seq, pos_labels, neg_labels = next(batch_iter)
        except StopIteration:
            break

        pos_seq = pos_seq.to(device)  # [B, max_len]
        neg_seq = neg_seq.to(device)
        pos_labels = pos_labels.to(device).float()  # [B, max_len], 1
        neg_labels = neg_labels.to(device).float()  # [B, max_len], 0

        # формируем общий вход (concat).
        # несколько вариантов — обучать Transformer на pos_seq и neg_seq раздельно или чередовать.
        # Делвем:
        #   1) делаем forward отдельно для pos_seq, получаем скрытые представления + head на айтем?
        #   2) тоже самое для neg_seq.
        #   3) считаем лосс.
        # Ниже дыа отдельных прохода.

        # Forward для pos_seq
        last_hidden_pos, _ = model(pos_seq)   # shape [B, max_len, emb]
        # SASRec предсказывает "следующий item" по последнему hidden,
        # но если хотим просто "точечно" получить логиты для самих pos_seq,
        # можно сделать скалярное произв.
        # Например, logits_pos[b, t] = dot( last_hidden_pos[b, t], E[item=t] ).
        # Но проще "head" сделать для каждого item:
        # в gSASRec по умолчанию get_predictions() смотрит на последний шаг.
        # Поэтрму, чтобы воспроизвести логику "по всем позициям", нужнр вычислять logits.
        # допустим: logits = W * hidden (linear), а itemID - часть лейбла.
        # Берем:
        # logits_pos[b,t] = < last_hidden_pos[b,t], output_embedding(pos_seq[b,t]) >

        output_embeddings = model.get_output_embeddings().weight  # shape [num_items+2, emb]
        # собираем вектор для каждого pos_seq[b,t]
        # pos_seq[b,t] — это индекс itemID (с учётом паддинга)
        # для удобства:
        bsz, seqlen = pos_seq.shape
        # создаем logits_pos[b, t] = dot( hidden[b,t], emb_of_item( pos_seq[b,t] ) )
        hidden_dim = last_hidden_pos.size(-1)
        emb_pos = output_embeddings[pos_seq.view(-1)]  # shape [B*seqlen, emb]
        hidden_pos = last_hidden_pos.view(-1, hidden_dim)  # same
        logits_pos = torch.sum(hidden_pos * emb_pos, dim=-1)  # [B*seqlen]
        logits_pos = logits_pos.view(bsz, seqlen)            # [B, seqlen]

        # то же самое для neg_seq
        last_hidden_neg, _ = model(neg_seq)
        emb_neg = output_embeddings[neg_seq.view(-1)]
        hidden_neg = last_hidden_neg.view(-1, hidden_dim)
        logits_neg = torch.sum(hidden_neg * emb_neg, dim=-1).view(bsz, seqlen)

        # итоговые logits и labels
        # посчитаем лосс отдельно и сложим.
        # L = L(pos) + L(neg).
        # mask = 1, если item != pad_value, иначе 0.
        pad_val = num_items + 1
        mask_pos = (pos_seq != pad_val).float()
        mask_neg = (neg_seq != pad_val).float()

        loss_pos_val = not_to_recommend_loss(logits_pos, pos_labels, mask_pos)
        loss_neg_val = not_to_recommend_loss(logits_neg, neg_labels, mask_neg)
        loss = loss_pos_val + loss_neg_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch}, step {step}, loss={loss_sum / (step)}")

    # валидация
    evaluation_result = evaluate(model, val_dataloader, config.metrics,
                                 config.recommendation_limit, config.filter_rated, device=device)
    print(f"Epoch {epoch} val_result: {evaluation_result}")

    metric_value = evaluation_result[config.val_metric]
    if metric_value > best_metric:
        best_metric = metric_value
        model_name = f"{models_dir}/gsasrec-{config.dataset_name}-epoch{epoch}-step{step}-val{best_metric:.4f}.pt"
        if best_model_name and os.path.exists(best_model_name):
            os.remove(best_model_name)
        best_model_name = model_name
        torch.save(model.state_dict(), model_name)
        steps_not_improved = 0
        print(f"New best model saved to {model_name}")
    else:
        steps_not_improved += 1
        if steps_not_improved >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break