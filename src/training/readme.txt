CHECKING THE ENVIRONMENT
To ensure the algorithm can run quickly, you will need to verify that the CUDA and CuDNN development kits are installed on the machine
1. type `nvcc --version` to check if CUDA is installed.
2. if not, stop the instance and find one that does have CUDA

WHERE TO PUT THE DATA
1. Download the .npz files containing data into a '/training_data' directory on the same level as the train_cnn.py script in the file heirarchy. These files cannot be within another directory inside '/training_data' (e.g. '/training_data/training/file.npz').
2. [IMPORTANT!] Move two of the files that are NOT augmented data out of the '/training_data' directory to be used as test data. Remember the names of these files - they can be placed in a new directory such as '/test_data' for to remember them for later.

RUNNING THE SCRIPT
Make sure to clone the repo containing everything and pull for changes before taking these steps to run it.
1. Run the script as a python script, preferrably from the '/src/training' directory (e.g. `python train_cnn.py` in the CLI). Python should already come installed and be added to the path.
2. Look at the console and be vigilant for any errors. Here are some potential problems:
  2.1 The console outputs 'cpu' instead of 'cuda'. This means that GPU isn't being used for training, which will make training times quite long. I would stop the instance there.
  2.2 There is an error regarding one of the imports - you may need to use pip to install some of the missing imports using `pip install <package>`
  2.3 The console hangs for an absurd amount of time (an hour+) when loading the data, or just outright crashes - there may not be enough RAM to hold the data and better hardware will be needed
  2.4 After the first batch starts to run, some error about GPU memory comes up. This has to do with the VRAM inside the GPU. Use `vim train_cnn.py` to go inside the script and decrease the total batch size to 16 or 8 (powers of 2).
3. Once you see the batches incrementing in the console, the script will most likely go off without a hitch. Keep an eye on it, but it should be guarenteed to execute without issue after the first fold is complete

GETTING THE MODELS AND FIGURES
Once the model is done executing, two new folders should appear: one called '/models' which contains .pth files with model data, and another '/figures' which contains images that visualize each model's performance.

Save these folders somewhere safe outside the instance (move inside S3, scp the files to your computer).

AFTERWARDS
Make sure to stop the instance when you are done using it.
