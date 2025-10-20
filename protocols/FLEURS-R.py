#!/usr/bin/env python
"""script to create protocol for FLEURS-R database
No additional protocol used, we simply walk through the directory,
no real audios in this database

/path/to/your/FLEURS/fleurs-r/data/
├── af_za/
│   ├── audio/
│   │   ├── dev/
│   │   │   ├── xx.wav
│   │   │   ├── . . .
│   │   ├── test/
│   │   │   ├── . . .
│   │   ├── train/
│   │   │   ├── . . .
├── am_et
├── . . .

FLEURS.csv:
"""
FLEURS_FOLDER_TO_SHORT = {
    "af_za": "AF",
    "am_et": "AM",
    "ar_eg": "AR",
    "as_in": "AS",
    "ast_es": "AST",
    "az_az": "AZ",
    "be_by": "BE",
    "bn_in": "BN",
    "bs_ba": "BS",
    "ca_es": "CA",
    "ceb_ph": "CEB",
    "cmn_hans_cn": "CN",
    "yue_hant_hk": "YUE_HANT",
    "cs_cz": "CS",
    "cy_gb": "CY",
    "da_dk": "DA",
    "de_de": "DE",
    "el_gr": "EL",
    "en_us": "EN",
    "es_419": "ES",
    "et_ee": "ET",
    "fa_ir": "FA",
    "ff_sn": "FF",
    "fi_fi": "FI",
    "fil_ph": "FIL",
    "fr_fr": "FR",
    "ga_ie": "GA",
    "gl_es": "GL",
    "gu_in": "GU",
    "ha_ng": "HA",
    "he_il": "HE",
    "hi_in": "HI",
    "hr_hr": "HR",
    "hu_hu": "HU",
    "hy_am": "HY",
    "id_id": "ID",
    "ig_ng": "IG",
    "is_is": "IS",
    "it_it": "IT",
    "ja_jp": "JA",
    "jv_id": "JV",
    "ka_ge": "KA",
    "kam_ke": "KAM",
    "kea_cv": "KEA",
    "kk_kz": "KK",
    "km_kh": "KM",
    "kn_in": "KN",
    "ko_kr": "KO",
    "ckb_iq": "CKB",
    "ky_kg": "KY",
    "lb_lu": "LB",
    "lg_ug": "LG",
    "ln_cd": "LN",
    "lo_la": "LO",
    "lt_lt": "LT",
    "luo_ke": "LUO",
    "lv_lv": "LV",
    "mi_nz": "MI",
    "mk_mk": "MK",
    "ml_in": "ML",
    "mn_mn": "MN",
    "mr_in": "MR",
    "ms_my": "MS",
    "mt_mt": "MT",
    "my_mm": "MY",
    "nb_no": "NB",
    "ne_np": "NE",
    "nl_nl": "NL",
    "nso_za": "NSO",
    "ny_mw": "NY",
    "oc_fr": "OC",
    "om_et": "OM",
    "or_in": "OR",
    "pa_in": "PA",
    "pl_pl": "PL",
    "ps_af": "PS",
    "pt_br": "PT",
    "ro_ro": "RO",
    "ru_ru": "RU",
    "bg_bg": "BG",
    "sd_in": "SD",
    "sk_sk": "SK",
    "sl_si": "SL",
    "sn_zw": "SN",
    "so_so": "SO",
    "sr_rs": "SR",
    "sv_se": "SV",
    "sw_ke": "SW",
    "ta_in": "TA",
    "te_in": "TE",
    "tg_tj": "TG",
    "th_th": "TH",
    "tr_tr": "TR",
    "uk_ua": "UK",
    "umb_ao": "UMB",
    "ur_pk": "UR",
    "uz_uz": "UZ",
    "vi_vn": "VI",
    "wo_sn": "WO",
    "xh_za": "XH",
    "yo_ng": "YO",
    "zu_za": "ZU"
}

import os
import sys
import csv
import glob

try:
    import pandas as pd
    import torchaudio
except ImportError:
    print("Please install pandas and torchaudio")
    sys.exit(1)


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Define paths
root_folder = '/path/to/your/'
dataset_name = 'FLEURS'
data_folder = os.path.join(root_folder, dataset_name, 'fleurs-r', 'data')
ID_PREFIX = 'FLEURS-R-'
output_csv = dataset_name + '-R.csv'

# Function to collect metadata from the directory structure
def collect_metadata(data_folder):
    metadata = []
    # List all wav files
    for file_path in sorted(
        glob.glob(os.path.join(data_folder, "**", "*.wav"), recursive=True)
    ):
        relative_path = file_path.replace(root_folder, "$ROOT/")
        # Extract relevant folder names
        parts = os.path.normpath(relative_path).split(os.sep)
# ['$ROOT', 'FLEURS', 'fleurs', 'data', 'ceb_ph', 'audio', 'train', '10587240206760547719.wav']
        lang_id = parts[4]
        lang = FLEURS_FOLDER_TO_SHORT[lang_id]
        subset = parts[6]
        if 'train' in subset:
            proportion = 'train'
        elif 'dev' in subset:
            proportion = 'valid'
        elif 'test' in subset:
            proportion = 'test'
        speaker = '-'
        label = 'fake'
        attack = '-'
        # ID
        file_id = f'{lang}-{os.path.splitext(parts[-1])[0]}'
        try:
            # Load metainfo with torchaudio
            metainfo = torchaudio.info(file_path)
            # Append metadata
            metadata.append({
                "ID": ID_PREFIX + file_id,
                "Label": label,
                "SampleRate": metainfo.sample_rate,
                "Duration": round(metainfo.num_frames / metainfo.sample_rate, 2), 
                "Path": relative_path,
                "Attack": attack,
                "Speaker": speaker,
                "Proportion": proportion,
                "AudioChannel": metainfo.num_channels,
                "AudioEncoding": metainfo.encoding,
                "AudioBitSample": metainfo.bits_per_sample,
                "Language": lang,
            })
        except Exception as e:
        # Handle any exception and skip this file
            print(f"Error: Could not load file {file_path}. Skipping. Reason: {e}")
    return metadata

# Write metadata to CSV
def write_csv(metadata):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata.to_csv(output_csv, index=False)

# Main script
if __name__ == "__main__":
    # Step 1: Collect metadata
    metadata = collect_metadata(data_folder)
    # Step 2: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
