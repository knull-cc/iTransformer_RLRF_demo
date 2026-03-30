export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom --features M \
  --seq_len 96 --pred_len 96 --label_len 0 \
  --e_layers 2 --enc_in 8 --dec_in 8 --c_out 8 \
  --d_model 128 --d_ff 128 --itr 1 \
  --w_point 1.7 --w_dir 0.0 --w_trend 0.3 --w_vol 0.3 --w_bias 0.05 --w_lag 0.1 --trend_window 7

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom --features M \
  --seq_len 96 --pred_len 192 --label_len 0 \
  --e_layers 2 --enc_in 8 --dec_in 8 --c_out 8 \
  --d_model 128 --d_ff 128 --itr 1 \
  --w_point 1.5 --w_dir 0.0 --w_trend 0.5 --w_vol 0.25 --w_bias 0.05 --w_lag 0.05 --trend_window 9

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom --features M \
  --seq_len 96 --pred_len 336 --label_len 0 \
  --e_layers 2 --enc_in 8 --dec_in 8 --c_out 8 \
  --d_model 128 --d_ff 128 --itr 1 \
  --w_point 1.3 --w_dir 0.0 --w_trend 0.7 --w_vol 0.2 --w_bias 0.05 --w_lag 0.0 --trend_window 11 \ 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom --features M \
  --seq_len 96 --pred_len 720 --label_len 0 \
  --e_layers 2 --enc_in 8 --dec_in 8 --c_out 8 \
  --d_model 128 --d_ff 128 --itr 1 \
  --w_point 1.1 --w_dir 0.0 --w_trend 0.9 --w_vol 0.15 --w_bias 0.05 --w_lag 0.0 --trend_window 13 \