from config import NTRSASRecExperimentConfig

config = NTRSASRecExperimentConfig(
    dataset_name='retailrocket',
    sequence_length=300,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.2,
    early_stopping_patience=200,
    max_epochs=10000,
)
