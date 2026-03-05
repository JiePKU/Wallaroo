cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json_parser;print json_parser.parse(\"$cluster_spec\", \"worker\")"
echo "worker list command is $worker_list_command"
eval worker_list=`python2 -c "$worker_list_command"`
echo "worker list is $worker_list"
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
index_command="import json_parser;print json_parser.parse(\"$cluster_spec\", \"index\")"
eval node_rank=`python2 -c "$index_command"`
echo "node rank is $node_rank"
dist_url="tcp://$master_addr:$master_port"
echo "dist url is $dist_url"
MASTER_ADDR=$master_addr
MASTER_PORT=8227
RANK=$node_rank
WORLD_SIZE=8
NPROC_PER_NODE=8

if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
	echo $NPROC_PER_NODE
	batch_size=40
	workers=24
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
	NPROC_PER_NODE=8
	batch_size=8
	workers=6
fi

DISTRIBUTED_ARGS="--nproc_per_node ${NPROC_PER_NODE} \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

cd /wallaroo
export PYTHONPATH=/wallaroo
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export WANDB_MODE=disabled
export http_proxy=http://10.70.18.69:8412
export https_proxy=http://10.70.18.69:8412

cp scripts/hacked_reader.py /workdir/conda_envs/wallaroo/lib/python3.10/site-packages/data_curation



# # Stage 1
# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /wallaroo/examples/wallaroo/ar_wallaroo_7/train_stage1.py \
#     config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage1_7B.yaml \


# # ## Stage 2  # important must train long enough
# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /wallaroo/examples/wallaroo/ar_wallaroo_7/train_stage2_and_stage3.py \
#     config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage2_7B.yaml \


# # ## Stage 3.1
# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /wallaroo/examples/wallaroo/ar_wallaroo_7/train_stage2_and_stage3.py \
#     config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage3_7B_512.yaml \


# ## Stage 3.2
# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /wallaroo/examples/wallaroo/ar_wallaroo_7/train_stage2_and_stage3.py \
#     config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage3_7B_mar_512.yaml \


## Stage 4
# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /wallaroo/examples/wallaroo/ar_wallaroo_7/train_stage4_omini.py \
#     config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage4_7B_omini.yaml \
