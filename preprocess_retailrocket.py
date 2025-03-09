import csv
from collections import defaultdict
import numpy as np
import json
from pathlib import Path

DATASET_DIR = Path('/content/datasets/retailrocket/events.csv').parent
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
RAW_FILE = DATASET_DIR / "events.csv"

"""
1) формируем для каждого пользователя последовательность позитивов (addtocart) и негативов (view)
2) Удаляем из негативов те itemid, которые у этого же user оказались в позитиве.
3) Отсеиваем товары (itemid), у которых <3 взаимодействий (view+addtocart).
4) Удаляем пользователей, у которых после этого <3 событий.
5) Переиндексируем itemid в диапазон [1..num_unique_items].
6) формируем на train/val/test:
   - ищем последний и предпоследний ПОЛОЖИТЕЛЬНЫЕ (addtocart) ивенты,
   - Остальные ивенты (включая негативные) до предпоследнего позитива => train,
   - Предпоследний позитив => val/output, всё, что до него => val/input,
   - Последний позитив => test/output, всё, что до него => test/input.
   Если у пользователя <2 позитивов, пропускаем такого пользователя.
7) Для train сохраняем раздельные файлы: input_pos.txt, input_neg.txt (только последовательность до val_idx).
   Для val/test – формат ml1m: input.txt, output.txt.
"""

def preprocess_and_split():
    DATASET_DIR.mkdir(exist_ok=True)
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)


    user_events = defaultdict(list)

    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            visitor_id = row['visitorid']
            event_type = row['event']
            item_id = row['itemid']
            timestamp = int(row['timestamp'])

            if event_type not in ['view', 'addtocart']:
                continue

            label = 1 if event_type == 'addtocart' else 0
            user_events[visitor_id].append((timestamp, item_id, label))

    # Шаг 1-2
    user_positive_items = defaultdict(set)
    for u, events in user_events.items():
        for (ts, it, lab) in events:
            if lab == 1:
                user_positive_items[u].add(it)

    for u, events in user_events.items():
        filtered = []
        for (ts, it, lab) in events:
            # если это негатив, но item встречался в позитиве => выкидываем
            if lab == 0 and it in user_positive_items[u]:
                continue
            filtered.append((ts, it, lab))
        user_events[u] = filtered

    # Шаг 3
    item_freq = defaultdict(int)
    for u, events in user_events.items():
        for (ts, it, lab) in events:
            item_freq[it] += 1

    valid_items = {it for it, f in item_freq.items() if f >= 3}

    # Шаг 4
    users_to_delete = []
    for u, events in user_events.items():
        filtered = [(ts, it, lab) for (ts, it, lab) in events if it in valid_items]
        if len(filtered) < 3:
            users_to_delete.append(u)
        else:
            user_events[u] = filtered

    for u in users_to_delete:
        del user_events[u]

    # Шаг 5
    all_items = set()
    for u, events in user_events.items():
        for (ts, it, lab) in events:
            all_items.add(it)
    all_items = sorted(all_items)
    new_id = {}
    for i, old in enumerate(all_items, start=1):
        new_id[old] = i

    for u in user_events:
        tmp = []
        for (ts, it, lab) in user_events[u]:
            tmp.append((ts, new_id[it], lab))
        user_events[u] = tmp


    num_sequences = 0
    num_clicks = 0
    num_purchase = 0

    # Шаг 6
    train_pos_sequences = []
    train_neg_sequences = []
    val_input = []
    val_output = []
    test_input = []
    test_output = []

    for u, events in user_events.items():
        events_sorted = sorted(events, key=lambda x: x[0])  # сорт по времени
        # индексы позитивов
        pos_indices = [i for i, e in enumerate(events_sorted) if e[2] == 1]
        if len(pos_indices) < 2:
            # если <2 позитивов, пропускаем
            continue

        # Предпоследний и последний позитив
        val_idx = pos_indices[-2]
        test_idx = pos_indices[-1]

        # train = события до val_idx
        train_evts = events_sorted[:val_idx]
        val_evt = events_sorted[val_idx]   # валидационный позитив
        test_evt = events_sorted[test_idx] # тестовый позитив

        val_input_events = events_sorted[:val_idx]
        # test_input = всё до test_idx
        test_input_events = events_sorted[:test_idx]

        all_user_events = events_sorted  # для подсчёта кликов/покупок
        num_sequences += 1
        for (_, _, lab) in all_user_events:
            if lab == 0:
                num_clicks += 1
            else:
                num_purchase += 1

        # формируем train_pos/train_neg
        train_pos_list = []
        train_neg_list = []
        for (ts, it, lab) in train_evts:
            if lab == 1:
                train_pos_list.append(it)
            else:
                train_neg_list.append(it)

        train_pos_sequences.append(train_pos_list)
        train_neg_sequences.append(train_neg_list)

        # val
        val_input.append([it for (ts, it, lab) in val_input_events])
        val_output.append(val_evt[1])

        # test
        test_input.append([it for (ts, it, lab) in test_input_events])
        test_output.append(test_evt[1])

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    with open(TRAIN_DIR / "input_pos.txt", "w") as f:
        for seq in train_pos_sequences:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(TRAIN_DIR / "input_neg.txt", "w") as f:
        for seq in train_neg_sequences:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "input.txt", "w") as f:
        for seq in val_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "output.txt", "w") as f:
        for it in val_output:
            f.write(str(it) + "\n")

    with open(TEST_DIR / "input.txt", "w") as f:
        for seq in test_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(TEST_DIR / "output.txt", "w") as f:
        for it in test_output:
            f.write(str(it) + "\n")

    num_items = len(all_items)
    num_interactions = num_clicks + num_purchase

    stats = {
        "num_users": num_sequences,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "#sequences": num_sequences,
        "#clicks (view)": num_clicks,
        "#purchase (addtocart)": num_purchase
    }
    print(json.dumps(stats, indent=4))

    with open(DATASET_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    preprocess_and_split()