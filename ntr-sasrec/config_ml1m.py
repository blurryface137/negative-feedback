from config import NTRSASRecExperimentConfig

config = NTRSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
)
