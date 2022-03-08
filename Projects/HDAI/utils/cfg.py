
class config:
    train_dsize = 900
    val_dsize = 100
    test_dsize = 100
    epoch = 100

    batch_size = 1
    val_batch_size = 2

    num_vote = 100
    
    smooth = 1e-9

    k = 1
    learning_rate = 1e-2
    lr_decay = 0.98

    log_dir = 'log'