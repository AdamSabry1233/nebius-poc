#!/bin/bash
# One command to set up the full environment
# Run: bash setup_env.sh

python3 -m venv ~/nebius-env
source ~/nebius-env/bin/activate

echo "Installing PyTorch with CUDA support..."
pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "Setup complete. Activate with: source ~/nebius-env/bin/activate"
