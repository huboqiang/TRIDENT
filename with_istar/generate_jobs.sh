#!/bin/bash
root='/cluster/home/panfy/projects/panlab/analysis/istar/'
code='/cluster/home/panfy/projects/panlab/code/istar/'  # 替换为实际路径
istar='/cluster/home/panfy/software/packages/istar/' # 替换为实际路径

n_genes=50
device="cuda"
pixel=0.274
patch=256

#   
# NCBI783

for sample in NCBI785 TENX95 TENX99 ; do
  for module in uni_v1 virchow gigapath conch_v1 conch_v15 ; do
    # 检查结果文件是否已存在
    prefix=$root'mliti-result/auto_ratio/'$sample'/'$module'/'
    if [ ! -d "$prefix" ]; then
      mkdir -p $prefix
    fi

    # 生成单个任务的SLURM提交脚本
    cat > '/cluster/home/panfy/projects/panlab/code/istar/sub_jobs/job_'$sample'_'$module'.sh' <<EOF
#!/bin/bash
#SBATCH --job-name=${sample}_${module}
#SBATCH --output=log_${sample}_${module}.out
#SBATCH --error=log_${sample}_${module}.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=300G

if [ -f "${root}HD_module_extracted_feature/${sample}/${module}/20x_${patch}px_0px_overlap/attention_features_${module}/${sample}_flatten.h5ad" ]; then
  cp ${root}he_pic/${sample}.tif ${prefix}he-raw.tif
  python ${code}step1.py ${prefix} --sample ${sample} --pixel ${pixel} --patch ${patch}
  echo 0.5 > ${prefix}pixel-size.txt
  python ${istar}rescale.py ${prefix} --image
  python ${istar}preprocess.py ${prefix} --image
  python ${istar}select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"
  python ${istar}rescale.py ${prefix} --locs --radius
  python ${code}step2.py ${prefix} --sample ${sample} --module ${module} --pixel ${pixel} --patch ${patch}
  python ${istar}impute.py ${prefix} --epochs=400 --device=${device}
  python ${code}step3.py ${prefix} --sample ${sample} --module ${module} --pixel ${pixel} --patch ${patch}
else
  echo "文件 ${sample}_${module}_${patch}_flatten.h5ad 不存在"
fi
EOF

    # 提交任务到集群
    sh '/cluster/home/panfy/projects/panlab/code/istar/sub_jobs/job_'$sample'_'$module'.sh'
    #nohup sh '/cluster/home/panfy/projects/panlab/code/istar/sub_jobs/job_'$sample'_'$module'.sh' > ${prefix}log_${sample}_${module}.out 2>&1 &
  done
done

# mkdir ${prefix}patch256
#   mv ${prefix}cnts-super ${prefix}patch256/${sample}_${module}_${patch}
#   python ${code}step2.py ${prefix} --sample ${sample} --module ${module} --pixel 0.5 --patch 467
#   python ${istar}impute.py ${prefix} --epochs=400 --device=${device}
#   python ${code}step3.py ${prefix} --sample ${sample} --module ${module} --pixel 0.5 --patch 467