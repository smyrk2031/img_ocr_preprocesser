from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
import os

# ------------------------------
# ユーザー設定（適宜変更）
# ------------------------------
image_path = "D:/winpythons/WPy64-31180_rinna/venv/donut/src/donut/test_iamges/table_c_01.png"

output_json_path = "output.json"       # 出力ファイルパス
model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"  # モデル名
task_prompt = "<s_cord-v2>"  # モデルに合わせてタスクプロンプトを変更

# ------------------------------
# モデルとプロセッサの読み込み
# ------------------------------
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.eval()

# ------------------------------
# 画像読み込みと前処理
# ------------------------------
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

# ------------------------------
# 推論実行
# ------------------------------
with torch.no_grad():
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)

# ------------------------------
# 結果のデコードと整形
# ------------------------------
decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
decoded_output = decoded_output.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").strip()

try:
    result_json = json.loads(decoded_output)
except json.JSONDecodeError:
    print("⚠️ JSONとしてパースできませんでした。結果をそのまま保存します。")
    result_json = {"raw_output": decoded_output}

# ------------------------------
# JSONとして保存
# ------------------------------
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False)

print(f"✅ 結果を '{output_json_path}' に保存しました。")
