from argparse import ArgumentParser
import os
import torch
from torchinfo import summary
from tqdm import tqdm
import random

from utils import load_config, build_model, get_device
from dataset_utils import get_num_items, get_train_dataloader, get_val_dataloader
from eval_utils import evaluate


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_rc15.py')
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

optimizer = torch.optim.Adam(model.parameters())
batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

# summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

def not_to_recommend_loss(logits, labels, mask):
    """
    logits: [B,S]
    labels: [B,S] (0,1,-1)
    mask:   [B,S], 1=учитывать, 0=пропуск
    label=1 => -log(sigmoid), label=0 => -log(1-sigmoid)
    """
    logits = logits.clamp(-10, 10)              # защита от экспоненциальных хвостов
    preds = torch.sigmoid(logits)
    eps = 1e-8
    
    pos_weight = 3.00
    neg_weight = 0.60

    pos_mask = (labels == 1).float() * mask
    neg_mask = (labels == 0).float() * mask

    loss_pos = -pos_weight * torch.log(preds + eps) * pos_mask
    loss_neg = -neg_weight * torch.log(1.0 - preds + eps) * neg_mask
    loss_elm = loss_pos + loss_neg
    
    denom = torch.sum(pos_mask + neg_mask) + eps
    return loss_elm.sum() / denom



# def not_to_recommend_loss(logits, labels, mask, base_pos_weight=1.0, base_neg_weight=0.3):
#     logits = logits.clamp(-10, 10)
#     preds = torch.sigmoid(logits)
#     eps = 1e-8

#     pos_mask = (labels == 1).float() * mask
#     neg_mask = (labels == 0).float() * mask

#     # Доля позитивов и негативов в текущем батче
#     num_pos = pos_mask.sum()
#     num_neg = neg_mask.sum()

#     # Адаптивные веса
#     if num_neg > 0:
#         neg_weight = base_neg_weight * (num_pos / num_neg)
#     else:
#         neg_weight = base_neg_weight

#     pos_weight = base_pos_weight

#     loss_pos = -pos_weight * torch.log(preds + eps) * pos_mask
#     loss_neg = -neg_weight * torch.log(1.0 - preds + eps) * neg_mask
#     loss_elm = loss_pos + loss_neg

#     denom = torch.sum(pos_mask + neg_mask) + eps
#     return loss_elm.sum() / denom


def negative_regularization(neg_out, weight=1e-3):
    # neg_out: [B,S,emb], хотим держать ближе к 0
    return (neg_out**2).mean() * weight


for epoch in range(config.max_epochs):
    model.train()
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0.0

    for batch_idx in pbar:
        step += 1
        try:
            items_batch, actions_batch = next(batch_iter)  # [B,S], [B,S], label=0,1 or -1
        except StopIteration:
            break

        items_batch = items_batch.to(device)
        actions_batch = actions_batch.to(device)

        # forward => seq_emb, neg_out
        inp_items = items_batch[:, :-1]
        inp_actions = actions_batch[:, :-1]
        tgt_items = items_batch[:, 1:]
        tgt_actions = actions_batch[:, 1:]
        
        seq_emb, neg_out, _ = model(inp_items, inp_actions)  # [B,S,emb], [B,S,emb]
        bsz, seqlen, emb_dim = seq_emb.shape

        # logits => dot(seq_emb[b,t], embed_of_item(items_batch[b,t]))
        out_emb = model.get_output_embeddings().weight   # [num_items+2, emb_dim]
        tgt_emb = out_emb[tgt_items]
        logits = torch.sum(seq_emb * tgt_emb, dim=-1)                 # B x (S-1)

        # mask => label != -1
        pad_mask = (tgt_actions == 1).float()             # ВОТ ЗДЕСЬ КОНЕЧНО БОЛЬШОЙ ВОПРОС ((tgt_actions == 0) | (tgt_actions == 1)).float()

        bce_loss = not_to_recommend_loss(logits, tgt_actions, pad_mask)
        reg_loss = negative_regularization(neg_out)
        total_loss = bce_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_sum += total_loss.item()
        wandb.log({"epoch": epoch + 1, "step": step, "train_loss": loss_sum})
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum/(batch_idx+1):.4f}")

    # Validation
    evaluation_result = evaluate(
        model,
        val_dataloader,
        config.metrics,
        config.recommendation_limit,
        config.filter_rated,
        device=device
    )
    metric_value = evaluation_result[config.val_metric]
    wandb.log({"epoch": epoch + 1, "val_metric": evaluation_result[config.val_metric]})
    print(f"Epoch {epoch} evaluation result: {evaluation_result}")

    # Сохранение лучшей модели
    if metric_value > best_metric:
        best_metric = metric_value
        model_name = (
            f"models/gsasrec-{config.dataset_name}-step:{step}"
            f"-emb:{config.embedding_dim}-dropout:{config.dropout_rate}"
            f"-val:{best_metric:.4f}.pt"
        )
        print(f"Saving new best model to {model_name}")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        steps_not_improved = 0
        torch.save(model.state_dict(), model_name)
    else:
        steps_not_improved += 1
        print(f"Validation metric did not improve for {steps_not_improved} steps")
        if steps_not_improved >= config.early_stopping_patience:
            print(f"Stopping training, best model was saved to {best_model_name}")
            break
