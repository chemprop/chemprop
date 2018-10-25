import os
# Which base dataset to train on
dataset = "descriptor_collection_MoDD4Q_2018.06.24_2018-09-24.csv_processed"

#Where to write these files too
output_dr = "/mmprojects/palmera2/Projects/MIT/chemprop/tmp"

# Files on remote computer
data_root = "/gpfs/nobackup/scratch/share/palmera2/modd4q/data/{}".format(dataset)
results_root = "/gpfs/nobackup/scratch/share/palmera2/modd4q/results/{}".format(dataset)

model_path = "/gpfs/nobackup/scratch/share/palmera2/modd4q/results/descriptor_collection_MoDD4Q_2018.06.24_2018-09-24.csv_processed/prio1.csv/graphconv/chemprop/model.best"
property_list = "Y:\palmera2\Projects\MoDD4Q\data\descriptors.csv"
models = ["chemprop", ]

data_files = [
    "HOMO_H2O.csv",
]

with open(property_list) as f:
    f.readline()
    f.readline()
    data_files = [fn + ".csv" for fn in f.readline().split(",")[3:-1]]

gpu_job_info = {
    "mem": "16GB",
    "walltime": "0:10:00",
    "ncpus": 1,
    "ngpus": 1
}

cpu_job_info = {
    "mem": "16GB",
    "walltime": "00:15:00",
    "ncpus": 20,
    "ngpus": 0
}

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

with open("master.sh", "wb") as f_master:
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
            with open(os.path.join("retrain_jobs", job_fn), "wb") as f_job:
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
                write_utf8(f_job, "python chemprop/retrain_test_regress.py"
                        + " --data_train {}".format(data_train_fn)
                        + " --data_test {}".format(data_test_fn)
                        + " --model_path {}".format(model_path)
                        + " --save_dir {}".format(results_folder)
                        + "\n")
                write_utf8(f_job, "date\n")
            f_master.write("qsub retrain_jobs/{}\n".format(job_fn).encode('utf-8'))
