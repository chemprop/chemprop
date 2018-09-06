import os
models = ["test",  "nn0", "nn1", "rfr", "svr", "lasso"]
gpu_models = ["nn0", "nn1"]
python_location = "python"
script_location = "ml_search.py"
root = "/gpfs/nobackup/scratch/share/palmera2/descriptors_collection_aug_dump_prio1/"
data_files = [root + f for f in [
    "dipol_H2O.csv",
    "H_HB_H2O.csv",
    "H_MF_H2O.csv",
    "ionization_potential_H2O.csv",
    "Moment_HBacc_H2O.csv",
    "mu_Cyclohexan.csv",
    "mu_Ethanol.csv",
    "mu_H2O.csv",
    "mu_solvent_H2O.csv",
    "polarizability_H2O.csv",
    "electron_affinity_H2O.csv",
    "H_int_H2O.csv"  ,
    "H_vdW_H2O.csv",
    "log_p_ow.csv",
    "Moment_HBdon_H2O.csv",
    "mu_DMSO.csv",
    "mu_Ethylacetat.csv" ,
    "mu_Octanol.csv",
    "mu_Triglyme.csv",]]

job_info = {
    "N": "",
    "l mem": "600 mb",
    "l walltime": "00:45:00",
    #"o": "path",
    #"e": "path"
}


def write_utf8(f, s):
    f.write(s.encode('utf-8'))


with open("master.sh", "wb") as f_master:
    f_master.write("#!/bin/bash\n".encode('utf-8'))
    for data_file in data_files:
        for model in models:
            job_info["N"] = model + "." + data_file.rsplit("/", 1)[1]
            job_fn = job_info["N"]+".job"
            with open(os.path.join("jobs", job_fn), "wb") as f_job:
                write_utf8(f_job, "#!/bin/bash\n")
                write_utf8(f_job, "#PBS -l select=1:ncpus=20:mem=10GB\n#PBS -l walltime=00:45:00\n")
                if model in gpu_models:
                    write_utf8(f_job, "#PBS -q gpuq\n")
                write_utf8(f_job, "module load rdkit/2017_09_2/intelpython3.6-2018.0\n")
                write_utf8(f_job, "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/app/intel/python3.6/2018.0/intelpython3/lib\n")
                write_utf8(f_job, "cd /gpfs/backup/users/home/palmera2/lib/modd4q\n")
                write_utf8(f_job, "source venv/bin/activate\n")
                write_utf8(f_job, "cd ml_search\n")
                write_utf8(f_job, "date +\"%m/%d/%Y %H:%M:%S $HOSTNAME\"\n")
                write_utf8(f_job, "python ml_search/ml_search.py --model \"{}\" --data_filename \"{}\"\n".format(model, data_file))
                write_utf8(f_job, "date +\"%m/%d/%Y %H:%M:%S $HOSTNAME\"\n")
            f_master.write("qsub jobs/{}\n".format(job_fn).encode('utf-8'))


