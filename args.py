import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Training-related hyperparameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait before early stop.')

    # Dataset-related hyperparameters
    parser.add_argument('--data_dir', type=str, default="./Data/",
                        help='Dataset location.')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')

    # GCN structure-related hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--step_num', type=int, default=2,
                        help='Number of message-passing steps')
    parser.add_argument('--nonlinear', dest='nonlinear', action='store_true')
    parser.add_argument('--linear', dest='nonlinear', action='store_false')
    parser.set_defaults(nonlinear=True)

    # Sampling-related hyperparameters
    parser.add_argument('--sample_scope', type=int, default=32,
                        help='Number of candidates for sampling')
    parser.add_argument('--sample_num', type=int, default=5,
                        help='Number of sampled neighbors')

    args, _ = parser.parse_known_args()
    return args
