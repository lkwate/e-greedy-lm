#!/bin/bash
#SBATCH --gres=gpu:2         # Number of GPUs (per node)
#SBATCH --cpus-per-task=4    # Number of CPUs
#SBATCH --mem=85G            # memory (per node)
#SBATCH --time=0-12:00       # time (DD-HH:MM)
#SBATCH --partition=main     # priority: unkillable > main > long
#SBATCH --job-name=KABROLG   #

module load cuda/10.1
source ../KABROLG_env/bin/activate

# Question Generation
filename=train-launcher-question-generation.sh
# Text Summarization
#filename=train-launcher-text-summarization.sh

chmod +x $filename
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

if [ $filename="train-launcher-question-generation.sh" ]; then
    . train-launcher-question-generation.sh
elif [ $filename="train-launcher-text-summarization.sh" ]; then
    . train-launcher-text-summarization.sh
fi

"""
############## README : Before runing this file on the cluster #################

module load python/3.7
virtualenv KABROLG_env
source KABROLG_env/bin/activate
pip install --upgrade pip

git clone https://github.com/lkwate/KABROLG
cd KABROLG
pip install -r requirements.txt
### for `import pytorch_lightning as pl` issues
pip3 install packaging
pip install importlib-metadata
pip install transformers -U
### for `from language_modelling import RLLMLightningModule` issues
pip3 install python-dateutil
pip uninstall attr
pip install attrs

tmux

#nano train-launcher-question-generation.sh
#nano train-launcher-text-summarization.sh

chmod +x cluster.sh

#
salloc --gres=gpu:2 -c 4 --mem=32Gb --time=12:00:00 --partition=main --job-name=KABROLG
. cluster.sh
# or 
srun --gres=gpu:2 -c 4 --mem=32Gb --time=12:00:00 --partition=main --job-name=KABROLG . cluster.sh
# or (see SBATCH parameters at the beginning of the file)
sbatch . cluster.sh
#
"""