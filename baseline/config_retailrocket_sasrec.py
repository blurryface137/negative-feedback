from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='retailrocket',
    sequence_length=10,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.2,
    negs_per_pos=1,
    gbce_t=0.0,
    early_stopping_patience=200,
    reuse_item_embeddings=False,
)
