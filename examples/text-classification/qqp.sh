export GLUE_DIR=my_glue_data
export GLUE_DIR=/home/jovyan/data-vol-2/zhengmeisong/data/qqp/part0TO9/
export GLUE_DIR=/home/jovyan/data-vol-1/zhengmeisong/wkspace/transformers/transformers/examples/text-classification/glue_data/
export TASK_NAME=QQP
outputDir=../../../py-models/${TASK_NAME}_e3/
mkdir $outputDir
cp $0 $outputDir
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 180 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --save_steps 10000 \
  --overwrite_output_dir \
  --output_dir ${outputDir} 2>&1 | tee ${outputDir}/log
