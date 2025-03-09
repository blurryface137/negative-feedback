from config import GSASRecExperimentConfig
from ir_measures import nDCG, R

config = GSASRecExperimentConfig(
    dataset_name='retailrocket',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=None,        #
    gbce_t = 0.0,
    reuse_item_embeddings=False,
    metrics=[nDCG@10, R@1, R@10],
    val_metric=nDCG@10,
    early_stopping_patience=20,
    max_epochs=10000
)