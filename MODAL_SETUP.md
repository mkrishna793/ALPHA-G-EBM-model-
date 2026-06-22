# Alpha-G: Modal Cloud Training Guide

This guide shows you exactly how to train the Alpha-G model on the 100k ARC Dataset using a Modal.com Jupyter Notebook.

## Step 1: Push to GitHub
1. Go to GitHub.com and create a new repository called `alpha-g`.
2. Push the local `D:\alpha-g` code to that repository:
   ```bash
   cd D:\alpha-g
   git init
   git add .
   git commit -m "Initial commit of Alpha-G"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/alpha-g.git
   git push -u origin main
   ```

## Step 2: Open your Modal Notebook
1. Log into your Modal account.
2. Spin up a new Jupyter Notebook.
3. **IMPORTANT:** When it asks you to select a GPU, select **NVIDIA H100** or **A100 (80GB)**.

## Step 3: Run the Training Code
Create a new cell in your Modal Notebook and paste exactly this:

```python
# 1. Download the Kaggle dataset
!pip install kaggle
# (Make sure your kaggle.json API key is uploaded to the notebook or set as environment variables!)
!mkdir kaggle_data
!kaggle datasets download -d arcgen100k/the-arc-gen-100k-dataset -p ./kaggle_data --unzip

# 2. Clone the Alpha-G architecture
!git clone https://github.com/YOUR-USERNAME/alpha-g.git

# 3. Install the package
%cd alpha-g
!pip install -e .

# 4. Train!
# This will automatically use the H100 Tensor Cores, BF16 precision, and Torch Compile.
!python src/alpha_g/train_kaggle.py
```

## Step 4: Download the Weights
When the training finishes (it should only take ~15 minutes on an H100), the script will save the trained model to `weights/alpha_g_kaggle.pth`. 

You can then download that `.pth` file from the Modal Notebook interface back to your local computer!
