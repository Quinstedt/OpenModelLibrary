To manage the project hours and storage there are some good commands to know such as: 

## Managing project budget
Check project status: `projinfo`

## Managing Jobs:
Run a job: `sbatch <filename or filepath>`

Check job info: `jobinfo -u <user>`

Check job queue: `squeue -u <user>`

Cancel job: `scancel <JobID> `( JobID can be found using the squeue command)

## Create job template
```
#!/usr/bin/env bash
#SBATCH -A NAISS 0000/00-000 -p alvis # NAISS 0000/00-000 is the project ID, in NAISS SUPR can be found as Dnr. 
#SBATCH --gpus-per-node=A100:3 # GPU type and number of GPUs. Here three A100 GPUs are used
#SBATCH -t 0-01:00:00  # The timeout limit for the job
#SBATCH -o job_output.out   # a file were we can track the terminal execution of the job

ml purge  # a good practice, to remove all activated modules

# running test - we cannot use alias. We execute the container and the commands needed to execute the file we want to run

apptainer exec /cephyr/NOBACKUP/groups/MyProject/Myfolder/mycontainer.sif python -m /cephyr/NOBACKUP/groups/MyProject/Myfolder/scr/pythonscript
```

