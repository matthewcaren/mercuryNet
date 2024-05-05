
import os
import json

root_dir = 'vids'
to_fix = ['_IZoa_VwMC0_60.0', 'NRhJhV4DBgw_222.84', 'kLfm-Eb1QR8_248.791', '1u9nECAmrOM_279.912967', 'ij0soRQpKhI_60.1', 'lozWDAsrWgw_227.688789', 'ewoDXsIaONA_60.033', 'UrRzvWEWaKU_235.033333', 'T4YdB2Caozs_128.4', '_IZoa_VwMC0_71.9', 'KgJUJxjea1s_142.683794']

for direc in to_fix:
    json_data = json.load(open(f'{root_dir}/{direc}/{direc}_feat.json'))
    json_data['lang'] = 'nolang'
    json.dump(json_data, open(f'{root_dir}/{direc}/{direc}_feat.json', 'w'))
