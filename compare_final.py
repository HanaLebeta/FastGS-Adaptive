import json
import os
import pandas as pd
import numpy as np

RESULTS_DIR = "output"        
SUFFIX = "_Adaptive"           

OUTPUT_EXCEL = "Adaptive_FastGS_Final_Results.xlsx"

baselines = {
    "bicycle":  [25.2673, 0.7553, 0.2456, 279.3],
    "flowers":  [21.6217, 0.6024, 0.3406, 261.3],
    "garden":   [27.5322, 0.8641, 0.1108, 448.1],
    "stump":    [27.1123, 0.7857, 0.2405, 204.7],
    "treehill": [22.8497, 0.6326, 0.3767, 206.0],
    "room":     [32.1534, 0.9304, 0.1890, 194.3],
    "counter":  [29.6019, 0.9180, 0.1767, 233.1],
    "kitchen":  [32.3622, 0.9392, 0.1044, 405.0],
    "bonsai":   [33.0393, 0.9535, 0.1599, 248.5],
}

scene_order = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
scene_types = ["Outdoor"] * 5 + ["Indoor"] * 4

def get_metrics(scene_name):
    folder = os.path.join(RESULTS_DIR, f"{scene_name}{SUFFIX}")
    json_path = os.path.join(folder, "results.json")
    time_path = os.path.join(folder, "time.txt")
    
    data = {}
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                js = json.load(f)
                key = "ours_30000"
                if key not in js: key = list(js.keys())[0]
                data['PSNR'] = js[key]['PSNR']
                data['SSIM'] = js[key]['SSIM']
                data['LPIPS'] = js[key]['LPIPS']
        except: return None
    else: return None

    if os.path.exists(time_path):
        with open(time_path, 'r') as f:
            try: data['Time'] = float(f.read().strip())
            except: data['Time'] = 0.0
    else: data['Time'] = 0.0
        
    return data

table_rows = []
ours_stats = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'Time': []}
base_stats = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'Time': []}

for scene, s_type in zip(scene_order, scene_types):
    base = baselines[scene]
    ours = get_metrics(scene)
    
    if ours:
        ours_stats['PSNR'].append(ours['PSNR'])
        ours_stats['SSIM'].append(ours['SSIM'])
        ours_stats['LPIPS'].append(ours['LPIPS'])
        if ours['Time'] > 0: ours_stats['Time'].append(ours['Time'])
        
        base_stats['PSNR'].append(base[0])
        base_stats['SSIM'].append(base[1])
        base_stats['LPIPS'].append(base[2])
        base_stats['Time'].append(base[3])

        table_rows.append({
            "Scene": scene,
            "Type": s_type,
            "Base PSNR": base[0],
            "Ours PSNR": ours['PSNR'],
            "Diff P": ours['PSNR'] - base[0],
            "Base SSIM": base[1],             
            "Ours SSIM": ours['SSIM'],        
            "Diff S": ours['SSIM'] - base[1], 
            "Base LPIPS": base[2],
            "Ours LPIPS": ours['LPIPS'],
            "Diff L": ours['LPIPS'] - base[2],
            "Base Time": base[3],
            "Ours Time": ours['Time'],
            "Speedup": base[3] / ours['Time'] if ours['Time'] > 0 else 0
        })

avg_o = {k: np.mean(v) for k, v in ours_stats.items()}
avg_b = {k: np.mean(v) for k, v in base_stats.items()}

table_rows.append({
    "Scene": "AVERAGE",
    "Type": "-",
    "Base PSNR": avg_b['PSNR'],
    "Ours PSNR": avg_o['PSNR'],
    "Diff P": avg_o['PSNR'] - avg_b['PSNR'],
    "Base SSIM": avg_b['SSIM'],             
    "Ours SSIM": avg_o['SSIM'],             
    "Diff S": avg_o['SSIM'] - avg_b['SSIM'],
    "Base LPIPS": avg_b['LPIPS'],
    "Ours LPIPS": avg_o['LPIPS'],
    "Diff L": avg_o['LPIPS'] - avg_b['LPIPS'],
    "Base Time": avg_b['Time'],
    "Ours Time": avg_o['Time'],
    "Speedup": avg_b['Time'] / avg_o['Time']
})

cols = [
    "Scene", "Type", 
    "Base PSNR", "Ours PSNR", "Diff P",
    "Base SSIM", "Ours SSIM", "Diff S", 
    "Base LPIPS", "Ours LPIPS", "Diff L",
    "Base Time", "Ours Time", "Speedup"
]

df = pd.DataFrame(table_rows)
df = df[cols]                       

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.to_string(index=False))
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"\n[Success] Results saved to {OUTPUT_EXCEL}")
