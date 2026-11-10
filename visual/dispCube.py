import matplotlib.pyplot as plt
import numpy as np

def plot_channels(matrix,cmap='viridis',save_name=None,use_bar=False,show=True):
    """
    Plot a 2D image for each channel of a 3D matrix, displayed in an n*n grid.
    If the number of channels is less than n*n, only the necessary subplots are used.

    Parameters:
    matrix -- a 3D numpy array with shape (C, H, W)
    n -- number of rows and columns in the subplot grid, should be an integer
    """

    # Get number of channels
    channels = matrix.shape[0]

    n = np.ceil(np.sqrt(channels)).astype(int)

    # Calculate rows and columns for subplots, ensuring not to exceed channel count
    n_rows = min(n, int(np.ceil(np.sqrt(channels))))
    n_cols = min(n, int(np.ceil(channels / n_rows)))

    # Create a new figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # If there is only one channel, axs is not an array, convert it to an array
    if channels == 1:
        axs = [axs]

    else:
        # Flatten axs to a 1D array for easy indexing
        axs = axs.flatten()

    # For each channel, plot a 2D image
    ims = []
    for i in range(channels):
        # Extract current channel data
        channel_data = matrix[i, :, :]

        # Plot the current channel data in the subplot
        im = axs[i].imshow(channel_data, cmap=cmap)
        ims.append(im)

        # Set title
        axs[i].set_title(f'Channel {i + 1}')

        # Hide x and y axis ticks
        axs[i].set_xticks([])
        axs[i].set_yticks([])


    # Hide extra subplots
    for i in range(channels, n_rows * n_cols):
        axs[i].axis('off')

    if use_bar:
        # Add a global colorbar, placed at the bottom center of the entire figure
        cbar = fig.colorbar(ims[n_cols], ax=axs, orientation='vertical',
                            pad=0.05, aspect=50, shrink=0.8)
        cbar.set_label('Pixel Value', fontsize=12)  # Add label for colorbar

    # Adjust spacing between subplots
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

def plot_list(matrix,cmap='viridis',n_rows=None,n_cols=None):
    """
    Plot each channel of a list as 2D images, displayed in an n*n grid.
    If the number of channels is less than n*n, only the necessary subplots are used.

    Parameters:
    matrix -- a list where each element has shape (H, W)
    n -- number of rows and columns in the subplot grid, should be an integer
    """

    # Get number of channels
    channels = len(matrix)

    n = np.ceil(np.sqrt(channels)).astype(int)

    # Calculate rows and columns for subplots, ensuring not to exceed channel count
    if n_rows is None:
        n_rows = min(n, int(np.ceil(np.sqrt(channels))))
        n_cols = min(n, int(np.ceil(channels / n_rows)))


    # Create a new figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # If there is only one channel, axs is not an array, convert it to an array
    if channels == 1:
        axs = [axs]

    else:
        # Flatten axs to a 1D array for easy indexing
        axs = axs.flatten()

    # For each channel, plot a 2D image
    for i in range(channels):
        # Extract current channel data
        channel_data = matrix[i][ :, :]

        # Plot the current channel data in the subplot
        axs[i].imshow(channel_data, cmap=cmap)

        # Set title
        axs[i].set_title(f'Channel {i + 1}')

        # Hide x and y axis ticks
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    # Hide extra subplots
    for i in range(channels, n_rows * n_cols):
        axs[i].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()
    # Display the plot
    plt.show()


if __name__ == '__main__':

    # Example usage
    # Create a random 3D matrix as an example
    C, H, W = 34, 300, 300  # e.g., 12 channels, 10x10 image
    matrix = np.random.rand(C, H, W)

    # Call the function to plot channels, assuming we want a 4x4 grid
    plot_channels(matrix)