python main_ssd.py --quantize --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --relu --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --relu --equalize --dataset_type voc12 --use_2007_metric false --log
# python main_ssd.py --relu --equalize --absorption --dataset_type voc12 --use_2007_metric false --log
# python main_ssd.py --quantize --relu --equalize --absorption --dataset_type voc12 --use_2007_metric false --log
# python main_ssd.py --relu --equalize --absorption --correction --dataset_type voc12 --use_2007_metric false --log
# python main_ssd.py --quantize --relu --equalize --absorption --correction --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --relu --equalize --correction --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --relu --correction --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --relu --correction --clip_weight --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --correction --dataset_type voc12 --use_2007_metric false --log
python main_ssd.py --quantize --correction --clip_weight --dataset_type voc12 --use_2007_metric false --log

# python main_ssd.py --quantize --dataset_type voc12 --use_2007_metric false --distill_range --log
# python main_ssd.py --quantize --relu --dataset_type voc12 --use_2007_metric false --distill_range --log
python main_ssd.py --quantize --relu --equalize --dataset_type voc12 --use_2007_metric false --distill_range --log
python main_ssd.py --quantize --relu --equalize --correction --dataset_type voc12 --use_2007_metric false --distill_range --log

python main_ssd.py --quantize --dataset_type voc07 --log
python main_ssd.py --quantize --relu --dataset_type voc07 --log
python main_ssd.py --quantize --relu --equalize --dataset_type voc07 --log
# python main_ssd.py --relu --equalize --absorption --dataset_type voc07 --log
# python main_ssd.py --quantize --relu --equalize --absorption --dataset_type voc07 --log
# python main_ssd.py --relu --equalize --absorption --correction --dataset_type voc07 --log
# python main_ssd.py --quantize --relu --equalize --absorption --correction --dataset_type voc07 --log
python main_ssd.py --quantize --relu --equalize --correction --dataset_type voc07 --log
python main_ssd.py --quantize --relu --correction --dataset_type voc07 --log
python main_ssd.py --quantize --relu --correction --clip_weight --dataset_type voc07 --log
python main_ssd.py --quantize --correction --dataset_type voc07 --log
python main_ssd.py --quantize --correction --clip_weight --dataset_type voc07 --log

# python main_ssd.py --quantize --dataset_type voc07 --distill_range --log
# python main_ssd.py --quantize --relu --dataset_type voc07 --distill_range --log
python main_ssd.py --quantize --relu --equalize --dataset_type voc07 --distill_range --log
python main_ssd.py --quantize --relu --equalize --correction --dataset_type voc07 --distill_range --log

