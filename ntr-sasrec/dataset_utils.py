import json
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    Получаем:
    input_file содержит строки вида:
      item1,label1 item2,label2 ...
    где label=0 или 1, а -1 не встречается.
    Для валидации/теста есть output_file (один item),
    а для train нет (output_file=None).

    Возвращаем:
     - inp_tensor: [max_len], где itemID
     - label_tensor: [max_len], где 0/1 или -1 (паддинг)
     - rated (set всех items) - если output_file есть
     - output item (int) - если output_file есть
    """

    def __init__(self, input_file, padding_value, output_file=None, max_length=200):
        self.inputs = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    self.inputs.append(([], []))
                    continue
                pairs = line.split()  # ["it1,lb1","it2,lb2",...]
                item_seq = []
                label_seq = []
                for p in pairs:
                    it_str, lb_str = p.split(',')
                    item_seq.append(int(it_str))
                    label_seq.append(int(lb_str))
                self.inputs.append((item_seq, label_seq))

        self.outputs = None
        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [int(line.strip()) for line in f]

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item_seq, label_seq = self.inputs[idx]
        if len(item_seq) > self.max_length:
            item_seq = item_seq[-self.max_length:]
            label_seq = label_seq[-self.max_length:]
        else:
            pad_len = self.max_length - len(item_seq)
            item_seq = [self.padding_value] * pad_len + item_seq
            # label=-1 для паддинга
            label_seq = [-1] * pad_len + label_seq

        inp_tensor = torch.tensor(item_seq, dtype=torch.long)
        lbl_tensor = torch.tensor(label_seq, dtype=torch.long)

        if self.outputs is not None:
            # val/test
            out_item = self.outputs[idx]
            rated = {it for it in item_seq if it <= self.padding_value - 1}
            return inp_tensor, lbl_tensor, rated, out_item
        else:
            # train
            return inp_tensor, lbl_tensor


def collate_train(batch):
    # batch — список кортежей (inp_tensor, lbl_tensor)
    items_list = []
    labels_list = []
    for (it, lbl) in batch:
        items_list.append(it)
        labels_list.append(lbl)
    items_batch = torch.stack(items_list, dim=0)  # [B,max_len]
    labels_batch = torch.stack(labels_list, dim=0)  # [B,max_len]
    return items_batch, labels_batch


def collate_val_test(batch):
    # batch — список (inp_tensor, lbl_tensor, rated, out_item)
    items_list = []
    labels_list = []
    rated_list = []
    out_list = []
    for (it, lbl, rd, ot) in batch:
        items_list.append(it)
        labels_list.append(lbl)
        rated_list.append(rd)
        out_list.append(ot)
    items_batch = torch.stack(items_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    out_batch = torch.tensor(out_list, dtype=torch.long)
    return items_batch, labels_batch, rated_list, out_batch


def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']


def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items'] + 1


def get_train_dataloader(dataset_name, batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    train_dataset = SequenceDataset(
        f"{dataset_dir}/train/input.txt",
        padding_value=padding_value,
        output_file=None,
        max_length=max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    return train_loader


def get_val_or_test_dataloader(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    ds = SequenceDataset(
        f"{dataset_dir}/{part}/input.txt",
        padding_value=padding_value,
        output_file=f"{dataset_dir}/{part}/output.txt",
        max_length=max_length
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_val_test)
    return loader


def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'val', batch_size, max_length)


def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'test', batch_size, max_length)
