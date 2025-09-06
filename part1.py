import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def naive_dft(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.
    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).
    Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
    """
    N = len(x)
    # Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)
    # Iterate through each frequency bin (k)
    for k in range(N):
        # For each frequency bin, sum the contributions from all input samples (n)
        for n in range(N):
            # The core DFT formula: x[n] * e^(-2j * pi * k * n / N)
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

def torch_dft(x, device):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal.
    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).
    Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
    """
    N = len(x)

    # Convert to tensor
    x = torch.tensor(x, dtype=torch.complex64, device=device)
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = n.view(N, 1) 
    # Build DFT matrix and multiply (O(N^2))
    W = torch.exp(-2j * torch.pi * (k @ n.view(1, N)) / N).to(torch.complex64)
    X = W @ x
    return X.detach().cpu().numpy()

if __name__ == "__main__":
    # 1. Generate the Signal
    # Parameters for the signal
    N = 600 # Number of sample points
    SAMPLE_RATE = 800.0 # Sampling rate in Hz
    FREQUENCY = 50.0 # Frequency of the sine wave in Hz
    # Calculate sample spacing
    T = 1.0 / SAMPLE_RATE

    # Create the time vector
    # np.linspace generates evenly spaced numbers over a specified interval.
    # We use endpoint=False because the interval is periodic.
    t = np.linspace(0.0, N * T, N, endpoint=False)

    # Create the sine wave signal
    y = np.sin(FREQUENCY * 2.0 * np.pi * t)

    # 2. Apply the DFT and Time the Execution
    # Time the naïve DFT implementation
    start_time_naive = time.time()
    dft_result = naive_dft(y)
    end_time_naive = time.time()
    naive_duration = end_time_naive - start_time_naive

    # Time NumPy's FFT implementation
    start_time_fft = time.time()
    fft_result = np.fft.fft(y)
    end_time_fft = time.time()
    fft_duration = end_time_fft - start_time_fft

    # Time torch DFT implementation
    start_time_fft = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fft_result = torch_dft(y, device)
    end_time_fft = time.time()
    torch_duration = end_time_fft - start_time_fft

    # 3. Print Timings and Verification
    print("--- DFT/FFT Performance Comparison ---")
    print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
    print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
    print(f"Torch DFT Execution Time: {torch_duration:.6f} seconds")
    # It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case
    if fft_duration > 0:
        print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")
    if torch_duration > 0:
        print(f"Torch DFT is approximately {naive_duration / torch_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")

    N = 10000 # Number of sample points
    SAMPLE_RATE = 800.0 # Sampling rate in Hz
    FREQUENCY = 50.0 # Frequency of the sine wave in Hz
    # Calculate sample spacing
    T = 1.0 / SAMPLE_RATE

    # Create the time vector
    # np.linspace generates evenly spaced numbers over a specified interval.
    # We use endpoint=False because the interval is periodic.
    t = np.linspace(0.0, N * T, N, endpoint=False)

    # Create the sine wave signal
    y = np.sin(FREQUENCY * 2.0 * np.pi * t)

    # 2. Apply the DFT and Time the Execution
    # Time the naïve DFT implementation
    start_time_naive = time.time()
    dft_result = naive_dft(y)
    end_time_naive = time.time()
    naive_duration = end_time_naive - start_time_naive

    # Time NumPy's FFT implementation
    start_time_fft = time.time()
    fft_result = np.fft.fft(y)
    end_time_fft = time.time()
    fft_duration = end_time_fft - start_time_fft

    # Time torch DFT implementation
    start_time_fft = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fft_result = torch_dft(y, device)
    end_time_fft = time.time()
    torch_duration = end_time_fft - start_time_fft

    # 3. Print Timings and Verification
    print("--- DFT/FFT Performance Comparison ---")
    print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
    print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
    print(f"Torch DFT Execution Time: {torch_duration:.6f} seconds")
    # It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case
    if fft_duration > 0:
        print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")
    if torch_duration > 0:
        print(f"Torch DFT is approximately {naive_duration / torch_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")

    N = 10 # Number of sample points
    SAMPLE_RATE = 800 # Sampling rate in Hz
    FREQUENCY = 50.0 # Frequency of the sine wave in Hz
    # Calculate sample spacing
    T = 1.0 / SAMPLE_RATE

    # Create the time vector
    # np.linspace generates evenly spaced numbers over a specified interval.
    # We use endpoint=False because the interval is periodic.
    t = np.linspace(0.0, N * T, N, endpoint=False)

    # Create the sine wave signal
    y = np.sin(FREQUENCY * 2.0 * np.pi * t)

    # 2. Apply the DFT and Time the Execution
    # Time the naïve DFT implementation
    start_time_naive = time.time()
    dft_result = naive_dft(y)
    end_time_naive = time.time()
    naive_duration = end_time_naive - start_time_naive

    # Time NumPy's FFT implementation
    start_time_fft = time.time()
    fft_result = np.fft.fft(y)
    end_time_fft = time.time()
    fft_duration = end_time_fft - start_time_fft

    # Time torch DFT implementation
    start_time_fft = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fft_result = torch_dft(y, device)
    end_time_fft = time.time()
    torch_duration = end_time_fft - start_time_fft

    # 3. Print Timings and Verification
    print("--- DFT/FFT Performance Comparison ---")
    print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
    print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
    print(f"Torch DFT Execution Time: {torch_duration:.6f} seconds")
    # It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case
    if fft_duration > 0:
        print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")
    if torch_duration > 0:
        print(f"Torch DFT is approximately {naive_duration / torch_duration:.2f} times faster.")
    else:
        print("FFT was too fast to measure a significant duration difference.")