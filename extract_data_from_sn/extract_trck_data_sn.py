#Exemple to extract tracking data from soccernet files.
# First, make sure to run the soccernet pipeline to get the tracking data output files.
# You can find the data in the folder 'outputs/sn-gamestate/date_YYYY-MM-DD/states/sn-gamestate.pklz'
# Download the file and the you can open and extract the data as shown below.

import zipfile
import pickle
import json
import pandas as pd
from pathlib import Path
from mplsoccer import Pitch
from matplotlib import pyplot as plt
BASE_DIR = Path(__file__).resolve().parent

#Example path to the zip file. (Game 027 and 028 from SN GSR dataset)
zip_path = BASE_DIR / "data_27_28"


with zipfile.ZipFile(zip_path, "r") as z:
    print("Contenu:", z.namelist())

    # 1) Lire le résumé JSON
    with z.open("summary.json") as f:
        summary = json.load(f)
    print("Résumé (clés):", list(summary.keys()))

    # 2) Charger le pickle principal 
    with z.open("027.pkl", "r") as f:
        data27 = pickle.load(f)
    print("type(data):", type(data27))

    # Charger l’autre pickle
    with z.open("028.pkl", "r") as f:
        data28 = pickle.load(f)
    print("type(data_img):", type(data28))
    
    



data28['x'] = data28['bbox_pitch'].apply(lambda x: x['x_bottom_middle'])
data28['y'] = data28['bbox_pitch'].apply(lambda x: x['y_bottom_middle'])


data28 = data28[["image_id","track_id","jersey_number","x","y","team"]]

data28['image_id'] = data28['image_id'].astype(int)
data28['jersey_number'] = data28['jersey_number'].fillna(0).astype(int)


jersey11_028 = data28[data28['track_id']==48]
jersey11_028.to_csv(BASE_DIR/"jersey11_028_trk.csv", index=False)



trxp = jersey11_028['x'] + 105.0 / 2.0
tryp = (-jersey11_028['y']) + 68.0 / 2.0
pitch = Pitch(pitch_type='custom', pitch_length=105.0, pitch_width=68.0)
fig, ax = pitch.draw(figsize=(8, 5))
ax.scatter(trxp, tryp, label='TRK (clean F4)', s=10, alpha=0.8)
ax.set_title("Player tracking data from SN-GSR (jersey 11 in game 028)")
ax.legend()
plt.savefig(BASE_DIR/"jersey11_028_trk.png", dpi=150)