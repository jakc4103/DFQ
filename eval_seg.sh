python main_seg.py --quantize --dataset voc12 --log
python main_seg.py --quantize --relu --dataset voc12 --log
python main_seg.py --quantize --relu --equalize --dataset voc12 --log
# python main_seg.py --relu --equalize --absorption --dataset voc12 --log
# python main_seg.py --quantize --relu --equalize --absorption --dataset voc12 --log
# python main_seg.py --relu --equalize --absorption --correction --dataset voc12 --log
# python main_seg.py --quantize --relu --equalize --absorption --correction --dataset voc12 --log
python main_seg.py --quantize --relu --equalize --correction --dataset voc12 --log
python main_seg.py --quantize --relu --correction --dataset voc12 --log
python main_seg.py --quantize --relu --correction --clip_weight --dataset voc12 --log
python main_seg.py --quantize --correction --dataset voc12 --log
python main_seg.py --quantize --correction --clip_weight --dataset voc12 --log

python main_seg.py --quantize --dataset voc12 --distill --log
python main_seg.py --quantize --relu --dataset voc12 --distill --log
python main_seg.py --quantize --relu --equalize --dataset voc12 --distill --log
python main_seg.py --quantize --relu --equalize --correction --dataset voc12 --distill --log

python main_seg.py --quantize --dataset voc07 --log
python main_seg.py --quantize --relu --dataset voc07 --log
python main_seg.py --quantize --relu --equalize --dataset voc07 --log
# python main_seg.py --relu --equalize --absorption --dataset voc07 --log
# python main_seg.py --quantize --relu --equalize --absorption --dataset voc07 --log
# python main_seg.py --relu --equalize --absorption --correction --dataset voc07 --log
# python main_seg.py --quantize --relu --equalize --absorption --correction --dataset voc07 --log
python main_seg.py --quantize --relu --equalize --correction --dataset voc07 --log
python main_seg.py --quantize --relu --correction --dataset voc07 --log
python main_seg.py --quantize --relu --correction --clip_weight --dataset voc07 --log
python main_seg.py --quantize --correction --dataset voc07 --log
python main_seg.py --quantize --correction --clip_weight --dataset voc07 --log

python main_seg.py --quantize --dataset voc07 --distill --log
python main_seg.py --quantize --relu --dataset voc07 --distill --log
python main_seg.py --quantize --relu --equalize --dataset voc07 --distill --log
python main_seg.py --quantize --relu --equalize --correction --dataset voc07 --distill --log
