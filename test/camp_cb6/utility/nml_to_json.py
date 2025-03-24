import os
import numpy as np
import pandas as pd

def to_json(lines, tol, phase='gas'):

    aero_data = pd.read_csv('aero_data.csv')
    aero_species = aero_data['Name'].str.strip(" '")
    aero_density = aero_data['Density']
    # aero_data.loc[aero_data['OM']=='F','korg'] = 0.5

    soa_defn = pd.read_csv('/home/dquevedo/AMBRS/jupyter_notebooks/soa_defn.csv')
    soa_species = list(soa_defn.loc[soa_defn['POA'].str.strip()=='F','PM Name'].str.strip(" '"))
    soag_species = soa_defn.loc[soa_defn['POA'].str.strip()=='F','Vapor Name'].str.strip(" '")
    soag_species.loc[soa_defn['Rxn Cntr Name'].str.strip(" '").str.len() > 0] = soa_defn.loc[soa_defn['Rxn Cntr Name'].str.strip(" '").str.len() > 0,'Rxn Cntr Name'].str.strip(" '")
    soag_species = list(soag_species)

    scales = pd.read_csv('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/ic_simple.dat', index_col=0, header=None)
    powers = [int(f'{ic[0]:.4e}'.split('e')[-1]) for ic in scales.values]
    tols = {name.strip(): 10**power for name, power in zip(scales.index, powers)}
    
    json = \
'''{
    "camp-data" : [
'''
    
    read = False
    last_line = False
    for line in lines:
        if '!SPECIES' in line:
            read = True
            continue
        elif '/' in line:
            last_line = True
            
        if read and not last_line:
            
            split_line = line.split(',')
            
            name = split_line[0].split("'")[1]
            if name in ['NUM','SRF',*soag_species,*soa_species]:
                continue
    
            molwt = split_line[1]

            if phase == 'aerosol':
                json += \
f'''        {{
            "name" : "{name}",
            "type" : "CHEM_SPEC",
            "phase" : "AEROSOL",
            "absolute integration tolerance" : {tol},
            "molecular weight [kg mol-1]" : {1.e-3*float(molwt):.4f},
            "density [kg m-3]" : {float(aero_density.loc[aero_species==name]):.4f},
            "num_ions" : {int(np.abs(aero_data.loc[aero_species==name,'Charge'])):n},
            "charge" : {int(aero_data.loc[aero_species==name,'Charge']):n},
            "kappa" : {float(aero_data.loc[aero_species==name,'korg']):.4f}
        }},
'''
            elif name == 'AH3OP':
                json += \
f'''        {{
            "name" : "{name}",
            "type" : "CHEM_SPEC",
            "phase" : "AEROSOL",
            "absolute integration tolerance" : {tol},
            "molecular weight [kg mol-1]" : {1.e-3*float(molwt):.4f},
            "density [kg m-3]" : {float(aero_density.loc[aero_species==name]):.4f},
            "num_ions" : 1,
            "charge" : 1,
            "kappa" : {float(aero_data.loc[aero_species==name,'korg']):.4f}
        }},
'''   
            elif name in ['SULF','SULRXN']:
                json += \
f'''        {{
            "name" : "{name}",
            "type" : "CHEM_SPEC",
            "absolute integration tolerance" : {tol},
            "molecular weight [kg mol-1]" : {1.e-3*float(molwt):.4f},
            "HLC(298K) [M Pa-1]" : 9.8692e5,
            "HLC exp factor [K]" : 8.684E+03,
            "diffusion coeff [m2 s-1]" : 1.260E-05,
            "notes" : "parameters from PartMC CAMP scenario"
        }},
'''
            elif name == 'SO2':
                json += \
f'''        {{
            "name" : "{name}",
            "type" : "CHEM_SPEC",
            "absolute integration tolerance" : {tol},
            "molecular weight [kg mol-1]" : {1.e-3*float(molwt):.4f},
            "diffusion coeff [m2 s-1]" : 1.3e-5,
            "notes" : "diff coeff from https://doi.org/10.1021/jp993622j"
        }},
'''
            else:
                json += \
f'''        {{
            "name" : "{name}",
            "type" : "CHEM_SPEC",
            "absolute integration tolerance" : {tol},
            "molecular weight [kg mol-1]" : {1.e-3*float(molwt):.4f}
        }},
'''
        if last_line:
            json = json[:-2] + json[-1]
            json += \
'''    ]
}
'''
    return(json)


file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/GC_cb6r5_ae7_aq.nml')

gc = file.readlines()

file.close()

file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/AE_cb6r5_ae7_aq.nml')

ae = file.readlines()

file.close()

file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/NR_cb6r5_ae7_aq.nml')

nr = file.readlines()

file.close()

tol = 1.0e-06
gc_json = to_json(gc, tol)
ae_json = to_json(ae, tol, phase='aerosol')
nr_json = to_json(nr, tol)

print(gc_json)
print(ae_json)
print(nr_json)

with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/gc_species.json','w') as json:
    json.write(gc_json)
with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/ae_species.json','w') as json:
    json.write(ae_json)
with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/nr_species.json','w') as json:
    json.write(nr_json)