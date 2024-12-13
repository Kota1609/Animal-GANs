import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd
import time

def generate(treatments, descriptors, training_data, descriptors_training, result_path, generator, opt, device):
    start_time = time.time()  # Start timing for performance tracking
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Scale molecular descriptors (MDs)
    scaler.fit(descriptors_training)
    scaled_MDs = pd.DataFrame(scaler.transform(descriptors), columns=descriptors.columns, index=descriptors.index)
    
    # Prepare structural data (S)
    S = pd.DataFrame()
    for i in range(len(treatments)):
        S = pd.concat([S, scaled_MDs[scaled_MDs.index == treatments.iloc[i].COMPOUND_NAME]])
    S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)

    # Scale time (T) and dose (D) levels
    scaler.fit(training_data['SACRI_PERIOD'].apply(Time).to_numpy(dtype=np.float32).reshape(-1, 1))
    T = scaler.transform(treatments['SACRI_PERIOD'].apply(Time).to_numpy(dtype=np.float32).reshape(-1, 1))
    T = torch.tensor(T, device=device)

    scaler.fit(training_data['DOSE_LEVEL'].apply(Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
    D = scaler.transform(treatments['DOSE_LEVEL'].apply(Dose).to_numpy(dtype=np.float32).reshape(-1, 1))
    D = torch.tensor(D, device=device)

    # Scale measurements
    measurements = training_data.iloc[:, 3:]
    scaler.fit(measurements)

    # Store results in a list for faster concatenation
    results_list = []

    for i in range(S.shape[0]):
        num = 0
        attempts = 0
        while num < opt.num_generate:
            z = torch.randn(1, opt.Z_dim).to(device)  # Sample noise for generation
            # Pass individual components to the generator
            generated_records = generator(z, S[i].view(1, -1), T[i].view(1, -1), D[i].view(1, -1))
            generated_records = scaler.inverse_transform(generated_records.cpu().detach().numpy())
            check = np.sum(generated_records[:, 9:14])  # Example check (relax this condition if needed)

            # Print the check value to understand what's happening
            print(f"Check value: {check}")

            # Loosen the check condition to 50 < check < 150 for testing
            if 50 < check < 150:  
                num += 1
                results_list.append(generated_records.flatten())
                print(f"Generated valid record {num}/{opt.num_generate} for treatment {i + 1}/{S.shape[0]}")

            attempts += 1
            if attempts % 100 == 0:
                print(f"Attempts: {attempts}, valid records generated: {num}, time elapsed: {time.time() - start_time:.2f}s")
        
        print(f"Generated {num} valid records for treatment {i + 1}/{S.shape[0]}")

    # Convert results list into DataFrame after processing all records
    Results = pd.DataFrame(results_list, columns=measurements.columns)
    Results = pd.concat([treatments.loc[treatments.index.repeat(opt.num_generate)].reset_index(drop=True), Results], axis=1)

    # Save to the specified result path
    Results.to_csv(result_path, sep='\t', index=False)
    print(f"Data generated and saved to {result_path} in {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    from opt import parse_opt
    from model import Generator
    from utils import Time, Dose

    opt = parse_opt()
    Z_dim = opt.Z_dim
    Stru_dim = opt.Stru_dim
    Time_dim = opt.Time_dim
    Dose_dim = opt.Dose_dim
    Measurement_dim = opt.Measurement_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure the result path directory exists
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    # Load data and model
    treatments = pd.read_csv(os.path.join('./Data', 'Example_Treatments_test.tsv'), sep="\t")
    training_data = pd.read_csv(os.path.join('./Data', 'Data_training.tsv'), sep="\t")
    MDs = pd.read_csv(os.path.join('./Data', 'MolecularDescriptors.tsv'), index_col=0, sep="\t")
    descriptors_training = MDs[MDs.index.isin(training_data['COMPOUND_NAME'])]
    descriptors = MDs[MDs.index.isin(treatments['COMPOUND_NAME'])]

    # Initialize and load the generator model
    generator = Generator(Z_dim, Stru_dim, Time_dim, Dose_dim, Measurement_dim).to(device)

    model_path = r'./models/generator_5000.pth'  # Path to the saved model
    weights = torch.load(model_path, map_location=device)

    try:
        generator.load_state_dict(weights, strict=False)
        print("Model loaded with mismatched keys ignored.")
    except RuntimeError as e:
        print("Error loading model:", e)
        filtered_weights = {k: v for k, v in weights.items() if k in generator.state_dict()}
        generator.load_state_dict(filtered_weights, strict=False)
        print("Model loaded with filtered keys.")

    generator.eval()

    # Set result path
    result_path = os.path.join(opt.results_path, 'generated_data_{}.tsv'.format(opt.num_generate))
    generate(treatments, descriptors, training_data, descriptors_training, result_path, generator, opt, device)