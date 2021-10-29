# ========================================
# simulation experiments
# ========================================
# step 1. generate random matrices
python experiments_simulation.py --step 1 --data-root <DATA_PATH>
# step 2. get results
python experiments_simulation.py --step 2



# ========================================
# ultra fine entity typing 
# ========================================
python experiments_text.py \
    --dataset ultrafine_entity_typing \
    --data-root=<DATA_PATH> --model-type denoise
python experiments_text.py \
    --dataset ultrafine_entity_typing --data-root=<DATA_PATH>


# ========================================
# relation extraction
# ========================================
python experiments_text.py \
    --dataset dialogue_re --data-root=<DATA_PATH>
python experiments_text.py \
    --dataset tacred --data-root=<DATA_PATH>


# ========================================
# imagenet experiments
# ========================================
# step 1: prepare the imagenet logits and targets for training and val set
python prepare_imagenet.py --out-dir <OUT_DIR> --data-root=<DATA_PATH>

# step 2: get results
python experiments_visual.py --dataset imagenet
