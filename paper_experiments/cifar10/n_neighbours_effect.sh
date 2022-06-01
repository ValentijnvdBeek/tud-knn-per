cd ../..

CHKPTS_PATH="chkpts/mobilenet-v2-cifar10.pth"

echo "=> generate data"

cd data/cifar10 || exit

rm -r all_data

python generate_data.py \
    --n_tasks 200 \
    --by_labels_split \
    --n_components -1 \
    --alpha 0.3 \
    --s_frac 1.0 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

for k in 1 3 5 7 10
do

  echo "Experiment with k = $k"

  python eval_knnper.py \
    cifar10 \
    random \
    $CHKPTS_PATH \
    $k \
    --capacities_grid_resolution 0.01 \
    --weights_grid_resolution 0.01 \
    --bz 256 \
    --device cuda \
    --verbose 1 \
    --results_dir "results/cifar10/n_neighbors_$k" \
    --seed 12345

  echo "-----------------------------------------"

done
