#!/bin/bash -l
#SBATCH --nodes=1             # 1 intel node
#SBATCH --ntasks=1          # 16 Cores
#SBATCH --mem-per-cpu=2G      # GB of RAM per CPU
#SBATCH --time=0:30:00       # HH:DD:SS
#SBATCH --output=slurm_%j.out # Standard output file
#SBATCH -p short
#SBATCH --job-name="Ansys Fluent Job"  # Name of Job
# The Journal file
#JOURNALFILE=FFF.jou
#OUTPUTFILE=output_cpu.log
TIMEFILE=time_cpu.log
# The Version of Fluent. The "dp" stands for "double precision"
#VERSION=3ddp
# Load samtools
module load ansys
# Number of Processors
#NTASKS=`echo $SLURM_TASKS_PER_NODE | cut -c1-2`
#NPROCS=`expr $SLURM_NNODES \* $NTASKS`
 START=$(date +%s.%N)
# Do work
#fluent -g $VERSION -i $JOURNALFILE >& $OUTPUTFILE
fluent -g 3ddp -i FFF.jou >& output_cpu.log
 END=$(date +%s.%N)
 DIFF=$(echo "$END - $START" | bc)
 echo $DIFF >& $TIMEFILE
~
