DATASETS=("cora" "citeseer" "pubmed" "amazon_computer" "amazon_photo" "ms_cs" "ms_physic")
SCOPES=(5 5 5 20 20 5 10)
SAMPLES=(1 1 1 1 1 1 1)

length=${#DATASETS[@]}

for ((i=1;i<=$length;i++))
do
    echo "DATASET: " ${DATASETS[$i]}
    python test.py --dataset "${DATASETS[$i]}" --sample_scope ${SCOPES[$i]} --sample_num ${SAMPLES[$i]}
done
