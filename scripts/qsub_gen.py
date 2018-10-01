import os
models = ["chemprop", ]
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

job_info = cpu_job_info

results_dir = "/gpfs/backup/users/home/palmera2/lib/chemprop/results/"

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

with open("master.sh", "wb") as f_master:
    f_master.write("#!/bin/bash\n".encode('utf-8'))
    for data_file in data_files:
        for model in models:
            job_name = model + "." + data_file.rsplit("/", 1)[1]
            if job_info["ngpus"] > 0:
                job_name += "_gpu"
            else:
                job_name += "_cpu"
            job_fn =job_name +".job"
            _results_dir = results_dir + job_name
            with open(os.path.join("jobs", job_fn), "wb") as f_job:
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
                write_utf8(f_job, "cd /gpfs/backup/users/home/palmera2/lib/chemprop/chemprop\n")
                write_utf8(f_job, "\n")
                write_utf8(f_job, "date +\" % m / % d / % Y % H: % M: % S $HOSTNAME\"\n")
                write_utf8(f_job, "python train_test_regress.py --data \"{}\" --save_dir \"{}\"\n".format(data_file,
                                                                                                          _results_dir))
                write_utf8(f_job, "date +\" % m / % d / % Y % H: % M: % S\"\n")
            f_master.write("qsub jobs/{}\n".format(job_fn).encode('utf-8'))
