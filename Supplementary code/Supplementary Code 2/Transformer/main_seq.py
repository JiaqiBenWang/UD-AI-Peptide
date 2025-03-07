import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from torch.utils.data import DataLoader
from utils_seq import make_data, MyDataSet, LinearWarmUpScheduler
from models_seq import Transformer
import argparse
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import _LRScheduler

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Regression', choices=['Regression'],
                    help='Task type.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2, help='Initial learning rate.')
parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps.')
parser.add_argument('--src_vocab_size', type=int, default=21, help='Number of amino acids.')
parser.add_argument('--src_len', type=int, default=4, help='Length of peptide sequences (set to 4 for tetrapeptides).')  # Default=4
parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'], help='Model type.')
parser.add_argument('--model_path', type=str, default='model_val_best.pt', help='Path to save the best model.')
args = parser.parse_args()

# Transformer model parameters setup
if args.model == 'Transformer':
    args.d_model = 512  # Embedding size
    args.d_ff = 2048  # FeedForward dimension
    args.d_k = args.d_v = 64  # Dimension of K(=Q), V
    args.n_layers = 6  # Number of Encoder layers
    args.n_heads = 8  # Number of heads in Multi-Head Attention

# Generate model path with model name, learning rate, and batch size to save the trained model
args.model_path = '{}_lr_{}_bs_{}.pt'.format(args.model, args.lr, args.batch_size)

# Set random seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

def main():
    if args.task_type == 'Regression':
        # Load data
        df_train = pd.read_csv(r'C:\Users\zzh\Desktop\Transformer\11500train_seqs.csv')
        df_valid = pd.read_csv(r'C:\Users\zzh\Desktop\Transformer\11500valid_seqs.csv')
        df_test = pd.read_csv(r'C:\Users\zzh\Desktop\test_data\test_data_10000.csv')

        # Prepare labels and features
        train_label = torch.Tensor(np.array(df_train["PI"])).unsqueeze(1).float()
        valid_label = torch.Tensor(np.array(df_valid["PI"])).unsqueeze(1).float().to(device)
        test_label = torch.Tensor(np.array(df_test["PI"])).unsqueeze(1).float().to(device)

        # Find the min and max values of the target for normalization
        args.max = train_label.max().item()
        args.min = train_label.min().item()

        train_feat = np.array(df_train["pep0"])
        valid_feat = np.array(df_valid["pep0"])
        test_feat = np.array(df_test["pep0"])

        # Prepare the inputs for the Transformer model
        train_enc_inputs = make_data(train_feat)
        valid_enc_inputs = make_data(valid_feat).to(device)
        test_enc_inputs = make_data(test_feat).to(device)

        # Create data loaders for training and validation
        train_loader = DataLoader(MyDataSet(train_enc_inputs, train_label), args.batch_size, shuffle=True)

        # Initialize variables for validation
        valid_rmse_saved = float('inf')  # Initialize as infinity to ensure the model is updated on the first validation
        loss_mse = torch.nn.MSELoss()

        # Initialize model and optimizer
        if args.model == 'Transformer':
            model = Transformer(args).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.0, momentum=0.9)

        # Initialize learning rate scheduler
        start_lr = 1e-3
        end_lr = args.lr
        warmup_steps = args.warmup_steps
        scheduler = LinearWarmUpScheduler(optimizer, warmup_steps, start_lr, end_lr)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            epoch_train_loss = 0
            for enc_inputs, labels in train_loader:
                enc_inputs = enc_inputs.to(device)
                labels = labels.to(device)
                outputs = model(enc_inputs)
                loss = loss_mse(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss to compute average loss later
                epoch_train_loss += loss.item()

            # Calculate average training loss for the epoch
            epoch_train_loss /= len(train_loader)

            # Update learning rate
            scheduler.step()

            # Validation
            if (epoch + 1) % 1 == 0:
                print(f'Starting validation for epoch {epoch + 1}')
                model.eval()
                with torch.no_grad():  # Disable gradient calculation for efficiency
                    try:
                        predict = model(valid_enc_inputs)
                        valid_mse = loss_mse(predict, valid_label).item()
                        valid_rmse = valid_mse ** 0.5  # Compute RMSE
                        print(f'Validation successful for epoch {epoch + 1}')
                    except Exception as e:
                        print(f'Error during validation for epoch {epoch + 1}: {e}')
                        continue

                    # Print RMSE for each epoch
                    print(f'Epoch: {epoch + 1}')
                    print(f'Valid Performance (RMSE): {valid_rmse}')

                    # Check if the RMSE is the best so far
                    if valid_rmse < valid_rmse_saved:
                        valid_rmse_saved = valid_rmse
                        print(f'New best RMSE: {valid_rmse}')
                        torch.save(model.state_dict(), args.model_path)

                    print(f"Epoch {epoch + 1} learning rate: {scheduler.get_last_lr()[0]}")

        # Testing
        predict = []
        model_load = Transformer(args).to(device)
        checkpoint = torch.load(args.model_path)
        model_load.load_state_dict(checkpoint)
        model_load.eval()

        with torch.no_grad():
            outputs = model_load(test_enc_inputs)
            predict = predict + outputs.squeeze(1).cpu().detach().numpy().tolist()
            labels = test_label.squeeze(1).tolist()

            # Prepare results for saving
            df_test_save = pd.DataFrame()
            file_path = r'C:\Users\zzh\Desktop\test_data\test_data_10000.csv'
            df_test_seq = pd.read_csv(file_path)
            df_test_save['pep0'] = df_test_seq['pep0']
            df_test_save['PI_predict'] = predict
            df_test_save['PI'] = labels

            # Calculate errors
            error = [labels[i] - predict[i] for i in range(len(labels))]
            absError = [abs(val) for val in error]
            squaredError = [val * val for val in error]
            MSE = sum(squaredError) / len(squaredError)
            RMSE = np.sqrt(MSE)
            MAE = sum(absError) / len(absError)
            R2 = r2_score(test_label.cpu(), outputs.cpu())

            df_test_save['MSE'] = squaredError
            df_test_save['MAE'] = absError
            df_test_save['MSE_ave'] = MSE
            df_test_save['RMSE_ave'] = RMSE
            df_test_save['MAE_ave'] = MAE
            df_test_save['R2'] = R2

            # # Save the Non-fixed test results as a CSV file
            # num_samples = os.path.basename(file_path).split('_')[2].split('.')[0]
            # df_test_save.to_csv(f'C:/Users/zzh/Desktop/Transformer/results_seq/{num_samples}logP_Test_reg_{args.model}_MAE_{MAE}_RMSE_{RMSE}_lr_{args.lr}_bs_{args.batch_size}.csv')

            # Save the fixed test results as a CSV file
            df_test_save.to_csv(
                f'C:/Users/zzh/Desktop/Transformer/results_seq/11500PI_Test_reg_{args.model}_MAE_{MAE}_RMSE_{RMSE}_lr_{args.lr}_bs_{args.batch_size}.csv')

        os.remove(args.model_path)  # Remove the model after testing

if __name__ == '__main__':
    main()
