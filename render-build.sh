
#!/usr/bin/env bash
set -e

echo "Installing Git LFS..."

curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz -o /tmp/git-lfs.tar.gz

tar -xzf /tmp/git-lfs.tar.gz -C /tmp

mkdir -p "$HOME/.local/bin"
cp /tmp/git-lfs-3.7.0/git-lfs "$HOME/.local/bin/git-lfs"
export PATH="$HOME/.local/bin:$PATH"

git lfs install --local

echo "Downloading model from Git LFS..."
git lfs pull

echo "Verifying model file..."
test -s models/baseline_rf_integrated_model.joblib

MODEL_SIZE=$(stat -c%s models/baseline_rf_integrated_model.joblib)

if [ "$MODEL_SIZE" -lt 1000000 ]; then
    echo "ERROR: Model file is too small. Git LFS download may have failed."
    exit 1
fi

echo "Model file verified: $MODEL_SIZE bytes"

echo "Installing API dependencies..."
pip install -r requirements-api.txt

echo "Build completed successfully."