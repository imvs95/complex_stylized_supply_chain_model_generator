#!/bin/bash -l                                                                                                       

#SBATCH --job-name="ReduceSize_Networks_40000"
#SBATCH --time=90:00:00
#SBATCH --partition=compute 

#SBATCH --nodes=1                                                                                                    
#SBATCH --exclusive                                                                                                  
#SBATCH --mem=0                                                                                                      
                                                                                     

#SBATCH --account=research-tpm-mas                                                                                   

module load 2022r2
module load openmpi
module load miniconda3


srun python remove_redudant_info_key.py