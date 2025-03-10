import json
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceWithNegDataset(Dataset):
    """
    Для train: храним два файла:
      1) input_pos.txt: каждая строка — последовательность позитивных itemid
      2) input_neg.txt: соответствующая строка — последовательность негативных itemid

    длины строк ghb этом должны совпасть по индексам
    или, как минимум, они соответствуют одному пользователю, но длины могут различаться

    В __getitem__(idx) возвращаем (pos_list, neg_list).
    """
    def __init__(self, pos_file, neg_file, padding_value, max_length=200):
        with open(pos_file, 'r') as f:
            self.pos_sequences = [list(map(int, line.strip().split())) for line in f]
        with open(neg_file, 'r') as f:
            self.neg_sequences = [list(map(int, line.strip().split())) for line in f]

        # проверяем, что pos и neg идут в том же порядке пользователей (одинаковое количество строк)
        assert len(self.pos_sequences) == len(self.neg_sequences), \
            "pos_file и neg_file должны иметь одинаковое число строк (одинаковое число пользователей)"

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.pos_sequences)

    def __getitem__(self, idx):
        pos_seq = self.pos_sequences[idx]
        neg_seq = self.neg_sequences[idx]
        # обрежем или дополним до max_length
        if len(pos_seq) > self.max_length:
            pos_seq = pos_seq[-self.max_length:]
        else:
            pos_seq = [self.padding_value]*(self.max_length - len(pos_seq)) + pos_seq

        if len(neg_seq) > self.max_length:
            neg_seq = neg_seq[-self.max_length:]
        else:
            neg_seq = [self.padding_value]*(self.max_length - len(neg_seq)) + neg_seq

        return torch.tensor(pos_seq, dtype=torch.long), torch.tensor(neg_seq, dtype=torch.long)


class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, max_length=200):
        with open(input_file, 'r') as f:
            self.inputs = [list(map(int, line.strip().split())) for line in f]

        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [int(line.strip()) for line in f]
        else:
            self.outputs = None

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        rated = set(inp)
        if len(inp) > self.max_length:
            inp = inp[-self.max_length:]
        else:
            inp = [self.padding_value]*(self.max_length - len(inp)) + inp

        inp_tensor = torch.tensor(inp, dtype=torch.long)

        if self.outputs is not None:
            out_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return inp_tensor, rated, out_tensor
        else:
            return (inp_tensor, )


def collate_val_test(batch):
    # для валидации/теста
    input_tensors = []
    rated_list = []
    output_tensors = []

    for x in batch:
        inp, rated, out = x
        input_tensors.append(inp)
        rated_list.append(rated)
        output_tensors.append(out)

    input_tensors = torch.stack(input_tensors, dim=0)
    output_tensors = torch.stack(output_tensors, dim=0)
    return [input_tensors, rated_list, output_tensors]


def not_to_recommend_collate(batch, pad_value):
    """
    коллатер для train
    формиуем общий батч
    Каждую позицию в pos_seq считаем позитивным примером (label=1),
    в neg_seq — негативным (label=0).
    Возвращаем: (batch_items, batch_labels),
    где batch_items.shape = [B, 2, max_len], batch_labels.shape = [B, 2, max_len].
    Или собираем их подряд. Здесь есть несколько вариантов;
    используем: "склеиваем по размерности seq_len" и храним отдельно mask.
    """

    # batch_pos.shape = [B, max_len], batch_neg.shape = [B, max_len]
    # labels_pos = ones, labels_neg = zeros
    pos_list = []
    neg_list = []
    for (pos_seq, neg_seq) in batch:
        pos_list.append(pos_seq)
        neg_list.append(neg_seq)

    pos_tensor = torch.stack(pos_list, dim=0)  # [B, max_len]
    neg_tensor = torch.stack(neg_list, dim=0)  # [B, max_len]

    # тензор меток
    pos_labels = torch.ones_like(pos_tensor)
    neg_labels = torch.zeros_like(neg_tensor)

    return pos_tensor, neg_tensor, pos_labels, neg_labels


def get_padding_value(dataset_name):
    with open(f"datasets/{dataset_name}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    padding_value = stats['num_items'] + 1
    return padding_value


def get_num_items(dataset_name):
    with open(f"datasets/{dataset_name}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']


def get_train_dataloader(dataset_name, batch_size=32, max_length=200):
    """
    Возвращаем DataLoader, который читает файлы input_pos.txt и input_neg.txt,
    и коллатит их в батчи с явными негативами.
    """
    dataset_dir = f"datasets/{dataset_name}"
    pad_val = get_padding_value(dataset_name)

    pos_file = f"{dataset_dir}/train/input_pos.txt"
    neg_file = f"{dataset_dir}/train/input_neg.txt"

    train_dataset = SequenceWithNegDataset(pos_file, neg_file, padding_value=pad_val, max_length=max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: not_to_recommend_collate(b, pad_val)
    )
    return train_loader


def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    pad_val = get_padding_value(dataset_name)
    inp_file = f"{dataset_dir}/val/input.txt"
    out_file = f"{dataset_dir}/val/output.txt"
    val_dataset = SequenceDataset(inp_file, pad_val, out_file, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return val_loader


def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    pad_val = get_padding_value(dataset_name)
    inp_file = f"{dataset_dir}/test/input.txt"
    out_file = f"{dataset_dir}/test/output.txt"
    test_dataset = SequenceDataset(inp_file, pad_val, out_file, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return test_loader
