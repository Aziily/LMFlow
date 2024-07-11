# Parses arguments
model_name_or_path=v2ray/stable-diffusion-3-medium-diffusers
arch_type=SD3transformer2D
dataset_path=data/example
preprocessor_kind=SD3
output_dir=output_dir
main_process_port=29501
img_size=1024

while [[ $# -ge 1 ]]; do
    key="$1"
    case ${key} in
        -m|--model_name_or_path)
            model_name_or_path="$2"
            shift
            ;;
        -t|--arch_type)
            arch_type="$2"
            shift
            ;;
        -d|--dataset_path)
            dataset_path="$2"
            shift
            ;;
        -k|--preprocessor_kind)
            preprocessor_kind="$2"
            shift
            ;;
        -o|--output_dir)
            output_dir="$2"
            shift
            ;;
        -p|--main_process_port)
            main_process_port="$2"
            shift
            ;;
        -i|--img_size)
            img_size="$2"
            shift
            ;;
        *)
            echo "error: unknown option \"${key}\"" 1>&2
            exit 1
    esac
    shift
done

echo "model_name_or_path: ${model_name_or_path}"
echo "arch_type: ${arch_type}"
echo "dataset_path: ${dataset_path}"
echo "output_dir: ${output_dir}"
echo "main_process_port: ${main_process_port}"
echo "img_size: ${img_size}"


accelerate launch \
    --config_file=configs/accelerate_t2i_sd3_config.yaml \
    --main_process_port=${main_process_port} \
    examples/finetune_t2i.py \
        --model_name_or_path=${model_name_or_path} \
        --arch_type=${arch_type} \
        --use_lora=False \
        --lora_target_module "to_k" "to_q" "to_v" "to_out.0" "add_k_proj" "add_v_proj" \
        --dataset_path=${dataset_path} \
        --preprocessor_kind=${preprocessor_kind} \
        --image_folder="img" \
        --image_size=${img_size} \
        --train_file="train.json" \
        --validation_file="valid.json" \
        --test_file="test.json" \
        --output_dir=${output_dir} \
        --logging_dir="logs" \
        --overwrite_output_dir=True \
        --mixed_precision="fp16" \
        --num_train_epochs=200 \
        --train_batch_size=1 \
        --learning_rate=1e-5 \
        --valid_steps=50
