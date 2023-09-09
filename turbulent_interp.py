import numpy as np
from scipy.fftpack import fftn, ifftn
import matplotlib.pyplot as plt

def generate_turbulent_field(N, seed=0):
    """Generate a turbulent field using the Fourier method."""
   # Step 1: Create a 3D grid of Fourier modes
    k1, k2, k3 = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), np.fft.fftfreq(N))

    # Calculate the magnitude of wave number k
    k = np.sqrt(k1**2 + k2**2 + k3**2)

    # Add a small number to k to avoid division by zero at [0,0,0]
    k[0, 0, 0] = 1e-10
    # Step 2: Generate the energy spectrum according to the Kolmogorov -5/3 law
    C = 10000000  # You would need to determine this constant appropriately
    E_k = C * k**(-11/3)
    # Avoid infinity at [0,0,0] for E_k
    E_k[0, 0, 0] = 0

    # Step 3: Generate complex-valued amplitudes
    np.random.seed(0)  # Set the seed for the random number generator
    random_phases = np.exp(1j * 2 * np.pi * np.random.random((N, N, N)))  # random phases between -1 and 1
    w_hat = np.sqrt(2 * E_k) * random_phases  # multiply by sqrt(2*E_k) to get amplitudes

    # Set w_hat[0,0,0]=0 to remove large background term in the velocity
    w_hat[0, 0, 0] = 0

    # Step 4: Calculate the velocity field at a given time
    w = np.real(ifftn(w_hat))
    # The 3D array `w` now represents a turbulent field
    return w


# The 3D array `w` now represents a turbulent field
#print(w.shape)

# Choose a slice to visualize
#w_slice = w[:, :, N//2]

# Create a grid of points
#X, Y = np.meshgrid(range(N), range(N))

# Plot the magnitude of the slice
#plt.imshow(np.abs(w_slice), origin='lower', cmap='viridis')
#plt.colorbar(label='Magnitude')

#plt.show()
