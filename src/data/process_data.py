import pandas as pd
import os

def process_and_save_data(input_file, output_folder):
    # Preberemo podatke iz CSV datoteke
    data = pd.read_csv(input_file)

    # Pretvorimo časovne žige v stolpcu "last_update" v časovni tip podatka
    data['last_update'] = pd.to_datetime(data['last_update'], unit='ms')
    
    # Zaokrožimo časovne žige na urni interval
    data['datetime'] = data['last_update'].dt.floor('H')

    # Združimo podatke po postajališčih in urah
    aggregated_data = data.groupby(['name', 'datetime']).agg({
        'available_bikes': 'mean',
        'available_bike_stands': 'mean'
    }).reset_index()

    # Seznam vseh postajališč
    postajalisca = aggregated_data['name'].unique()

    for postajalisce in postajalisca:
        # Filtriramo podatke za trenutno postajališče
        filtered_data = aggregated_data[aggregated_data['name'] == postajalisce]
        
        # Ime datoteke za shranjevanje podatkov
        ime_datoteke = os.path.join(output_folder, f"{postajalisce.replace(' ', '_')}.csv")
        
        # Shranimo filtrirane podatke v CSV datoteko
        # z načinom 'w' za pisanje, da se podatki prepišejo
        filtered_data.to_csv(ime_datoteke, mode='w', index=False)

        print(f"Podatki za postajališče {postajalisce} so bili uspešno prepisani v datoteko {ime_datoteke}")

# Uporaba funkcije
input_file = '../data/raw/fetch_mbajk.csv'
output_folder = '../data/processed'
process_and_save_data(input_file, output_folder)
