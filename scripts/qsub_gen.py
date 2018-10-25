import os

# Which base dataset to train on
dataset = "descriptor_collection_MoDD4Q_2018.06.24_2018-09-24.csv_processed"

#Where to write these files too
output_dr = "/mmprojects/palmera2/Projects/MIT/chemprop/tmp"

# Files on remote computer
data_root = "/gpfs/nobackup/scratch/share/palmera2/modd4q/data/{}".format(dataset)
results_root = "/gpfs/nobackup/scratch/share/palmera2/modd4q/results/{}".format(dataset)

# Individual training tasks for properties in this dataset
data_files = [
    "HOMO_H2O.csv",
    "LUMO_H2O.csv",
    "Moment_HBacc_H2O.csv",
    "ionization_potential_H2O.csv",
    "dipol_H2O.csv",
    "H_HB_H2O.csv",
    "H_MF_H2O.csv",
    "polarizability_H2O.csv",
    "electron_affinity_H2O.csv",
    "H_int_H2O.csv" ,
    "H_vdW_H2O.csv",
    "log_p_ow.csv",
    "Moment_HBdon_H2O.csv",
    "mu_Cyclohexan.csv",
    "mu_Ethanol.csv",
    "mu_H2O.csv",
    "mu_solvent_H2O.csv",
    "mu_DMSO.csv",
    "mu_Ethylacetat.csv" ,
    "mu_Octanol.csv",
    "mu_Triglyme.csv",
    "prio1.csv",
    "pca100.csv"
    ]

data_files = [
     "pca100.csv"
    ]

gpu_job_info = {
    "mem": "16GB",
    "walltime": "4:00:00",
    "ncpus": 1,
    "ngpus": 1
}

cpu_job_info = {
    "mem": "16GB",
    "walltime": "24:00:00",
    "ncpus": 40,
    "ngpus": 0
}

models = ["chemprop", ]

job_info = cpu_job_info

def write_utf8(f, s):
    f.write(s.encode('utf-8'))

def select_str(job_info):
    s = "#PBS -l select=1"
    if job_info['ngpus'] > 0:
        s += ":ngpus=1"
    s += ":ncpus={}".format(job_info["ncpus"])
    s += ":mem={}".format(job_info["mem"])
    s += "\n"
    return s

with open(os.path.join(output_dr, "master.sh"), "wb") as f_master:
    f_master.write("#!/bin/bash\n".encode('utf-8'))
    for data_file in data_files:
        data_train_fn = data_root + "/train/" + data_file
        data_test_fn = data_root + "/test/" + data_file
        for model in models:
            job_name = model + "." + data_file
            if job_info["ngpus"] > 0:
                job_name += "_gpu"
            else:
                job_name += "_cpu"
            job_fn =job_name + ".job"
            results_folder = results_root + "/" + data_file + "/graphconv/" + model
            with open(os.path.join(output_dr, "jobs", job_fn), "wb") as f_job:
                write_utf8(f_job, "#!/bin/bash\n")
                write_utf8(f_job, select_str(job_info))
                write_utf8(f_job, "#PBS -l walltime={}\n".format(job_info["walltime"]))
                write_utf8(f_job, "#PBS -N {}\n".format(job_fn))
                if job_info["ngpus"] > 0:
                    write_utf8(f_job, "#PBS -q gpuq\n")
                write_utf8(f_job, "\n")
                write_utf8(f_job, "module load rdkit/2017_09_2/intelpython2.7-2018.1\n")
                write_utf8(f_job,
                           "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/app/intel/python2.7/2018.1/intelpython2/lib\n")
                write_utf8(f_job, "source /gpfs/backup/users/home/palmera2/lib/chemprop/venv/bin/activate\n")
                write_utf8(f_job, "cd /gpfs/backup/users/home/palmera2/lib/chemprop/\n")
                write_utf8(f_job, "$HOSTNAME\n")
                write_utf8(f_job, "date\n")
                write_utf8(f_job, "python chemprop/train_test_regress.py"
                        + " --data_train {}".format(data_train_fn)
                        + " --data_test {}".format(data_test_fn)
                        + " --save_dir {}".format(results_folder)
                        + "\n")
                write_utf8(f_job, "date\n")
            f_master.write("qsub jobs/{}\n".format(job_fn).encode('utf-8'))
