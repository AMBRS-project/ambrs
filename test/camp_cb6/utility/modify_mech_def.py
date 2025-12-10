import os
import pandas as pd

def to_json(lines):

    soa_defn = pd.read_csv('/home/dquevedo/AMBRS/jupyter_notebooks/soa_defn.csv')
    soa_species = list(soa_defn.loc[soa_defn['POA'].str.strip()=='F','PM Name'].str.strip(" '"))
    soag_species = soa_defn.loc[soa_defn['POA'].str.strip()=='F','Vapor Name'].str.strip(" '")
    soag_species.loc[soa_defn['Rxn Cntr Name'].str.strip(" '").str.len() > 0] = soa_defn.loc[soa_defn['Rxn Cntr Name'].str.strip(" '").str.len() > 0,'Rxn Cntr Name'].str.strip(" '")
    soag_species = list(soag_species)
    
    j_rates = pd.read_csv('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/phot_j_labeled.dat', header=None, index_col=0)

    running_rates = {}
    
    json = \
'''{
    "camp-data" : [
        {
            "type" : "RELATIVE_TOLERANCE",
            "value" : 1.0e-04
        },
                {
                    "type" : "SIMPOL_PHASE_TRANSFER",
                    "gas-phase species" : "SOAG",
                    "aerosol phase" : "mixed",
                    "aerosol-phase species" : "SOA",
                    "B" : [ -5813.039000, 7.953364, -0.010543, 0.995233 ],
                    "notes" : "Representative of pinic acid"
                },
                {
                    "type" : "SIMPOL_PHASE_TRANSFER",
                    "gas-phase species" : "SULF",
                    "aerosol phase" : "mixed",
                    "aerosol-phase species" : "H2SO4_aq",
                    "aerosol-phase water" : "AH2O",
                    "B" : [ 0.0, -7.1078, 0.0, 0.0 ],
                    "notes" : "B2 = log10{VP[atm]} (H2SO4 vapor pressure from PubChem) for SIMPOL"
                },
                {
                    "type" : "SIMPOL_PHASE_TRANSFER",
                    "gas-phase species" : "SULRXN",
                    "aerosol phase" : "mixed",
                    "aerosol-phase species" : "H2SO4_aq",
                    "aerosol-phase water" : "AH2O",
                    "B" : [ 0.0, -7.1078, 0.0, 0.0 ],
                    "notes" : "B2 = log10{VP[atm]} (H2SO4 vapor pressure from PubChem) for SIMPOL"
                },
                {
                    "type" : "AQUEOUS_EQUILIBRIUM",
                    "aerosol phase" : "mixed",
                    "aerosol-phase water" : "AH2O",
                    "A" : 1.0e+3,
                    "C" : 0.0,
                    "k_reverse" : 5.0e+10,
                    "ion pair" : "H-HSO4",
                    "reactants" : {
                        "H2SO4_aq" : {}
                    },
                    "products" : {
                        "HSO4_aq" : {},
                        "AH3OP" : {}
                    }
                },
                {
                    "type" : "AQUEOUS_EQUILIBRIUM",
                    "aerosol phase" : "mixed",
                    "aerosol-phase water" : "AH2O",
                    "A" : 1.02e-2,
                    "C" : 2.7e+3,
                    "k_reverse" : 1.0e+11,
                    "ion pair" : "H-SO4",
                    "reactants" : {
                        "HSO4_aq" : {}
                    },
                    "products" : {
                        "ASO4" : {},
                        "AH3OP" : {}
                    }
                },
'''

    oneliner = ''.join([line.strip() for line in lines])
    lines = oneliner.split(';')
    
    read = False
    last_line = False
    rxn_id = 1 # n emission rxns
    phot_id = 0
    for line in lines:
        if 'REACTIONS[CM]' in line:
            read = True
            line = '='.join(line.split('=')[1:])
        elif 'END MECH' in line:
            last_line = True
            
        if read and not last_line:

            if 'POA_AGE' in line or 'RPOAGE' in line or 'OLIG' in line or 'HYD_MT' in line:
                continue

            rxn = line
            rxn_type = None

            # if rxn[0] == '!':
            #     continue
            if 'R92a' in rxn or 'R92b' in rxn or 'R215' in rxn or 'CL29' in rxn:
                continue
                
            if '&' in rxn:
                if '%2' in rxn:
                    rxn_type = 'CMAQ_OH_HNO3'
                elif '%3' in rxn:
                    rxn_type = 'CMAQ_H2O2'
                elif '%H' not in rxn:
                    rxn_type = 'TROE'
            elif '@' in rxn or ('@' not in rxn and '&' not in rxn and '*K' not in rxn and '/<' not in rxn and '~' not in rxn) :
                rxn_type = 'ARRHENIUS'
            elif '/<' in rxn:
                rxn_type = 'PHOTOLYSIS'
            elif 'HETERO' in rxn:
                # rxn_type = 'HETEROGENEOUS'
                rxn_type = None
            elif 'K' in rxn:
                rxn_type = 'REFERENCE'

            if rxn_type != None:
                rxn_id += 1

            if rxn_type == 'CMAQ_OH_HNO3':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('%2 #')
                k0, k2, k3 = rate.split('&')
                A0, C0 = k0.split('@')
                A2, C2 = k2.split('@')
                A3, C3 = k3.split('@')
                parameters = \
f'''
                    "k0_A" : {float(A0):.4e},
                    "k0_B" : {0.0000},
                    "k0_C" : {-float(C0):.4e},
                    "k2_A" : {float(A2):.4e},
                    "k2_B" : {0.0000},
                    "k2_C" : {-float(C2):.4e},
                    "k3_A" : {float(A3):.4e},
                    "k3_B" : {0.0000},
                    "k3_C" : {-float(C3):.4e},
'''
            if rxn_type == 'CMAQ_H2O2':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('%3 #')
                k1, k2 = rate.split('&')
                A1, C1 = k1.split('@')
                A2, C2 = k2.split('@')
                parameters = \
f'''
                    "k1_A" : {float(A1):.4e},
                    "k1_B" : {0.0000},
                    "k1_C" : {-float(C1):.4e},
                    "k2_A" : {float(A2):.4e},
                    "k2_B" : {0.0000},
                    "k2_C" : {-float(C2):.4e},
'''
            if rxn_type == 'TROE':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                and_split = rate.split('&')
                if len(and_split) == 2:
                    N = 1.0
                    Fc = 0.6
                    k0, kinf = and_split
                elif len(and_split) == 3:
                    Fc = 0.6
                    k0, kinf, N = and_split
                else:
                    k0, kinf, N, Fc = and_split
                at_split = k0.split('@')
                if len(at_split) > 1:
                    A0B0, C0 = at_split
                else:
                    C0 = 0.0
                    A0B0 = at_split[0]
                carat_split = A0B0.split('^')
                if len(carat_split) > 1:
                    A0, B0 = carat_split
                else:
                    B0 = 0.0
                    A0 = carat_split[0]
                at_split = kinf.split('@')
                if len(at_split) > 1:
                    AinfBinf, Cinf = at_split
                else:
                    Cinf = 0.0
                    AinfBinf = at_split[0]
                carat_split = AinfBinf.split('^')
                if len(carat_split) > 1:
                    Ainf, Binf = carat_split
                else:
                    Binf = 0.0
                    Ainf = carat_split[0]
                parameters = \
f'''
                    "k0_A" : {float(A0):.4e},
                    "k0_B" : {float(B0):.4e},
                    "k0_C" : {-float(C0):.4e},
                    "kinf_A" : {float(Ainf):.4e},
                    "kinf_B" : {float(Binf):.4e},
                    "kinf_C" : {-float(Cinf):.4e},
                    "Fc" : {float(Fc):.4e},
                    "N" : {float(N):.4e},
'''
            if rxn_type == 'ARRHENIUS':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                ABC = rate.split('@')
                if len(ABC) > 1:
                    AB, C = ABC
                else:
                    AB = ABC[0]
                    C = 0.
                AB_split = AB.split('^')
                if len(AB_split) > 1:
                    A, B = AB_split
                else:
                    if AB.strip() == '8.3-11':
                        AB = '8.3E-11'
                    A = AB
                    B = 0.
                parameters = \
f'''
                    "A" : {float(A):.4e},
                    "B" : {float(B):.4e},
                    "C" : {-float(C):.4e},
'''
            if rxn_type == 'PHOTOLYSIS':
                phot_id += 1
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                parameters = \
f'''
                    "base rate" : {float(j_rates.loc[rate.split('/')[-1][1:-1]]):.4e},
                    "notes" : "{rate}",
'''

            if rxn_type == 'REFERENCE':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                rate_split = rate.split('*K')
                scale = rate_split[0]
                ref = running_rates[f'{rate_split[1]}']
                rxn_type = ref['rxn_type']
                parameters = ref['parameters']
                if rxn_type == 'ARRHENIUS':
                    A = float(scale)*float(ref['param'])
                    if '"A_ref" :' in parameters:
                        parameters = parameters.replace('"A" :', '"A_ref2" :')
                        parameters = parameters.replace('"A_ref" :', '"A_ref1" :')
                    else:
                        parameters = parameters.replace('"A" :', '"A_ref" :')
                    parameters += \
f'''
                    "A" : {A:.4e},
'''
                elif rxn_type == 'TROE':
                    Fc = float(scale)*float(ref['param'])
                    if '"Fc_ref" :' in parameters:
                        parameters = parameters.replace('"Fc" :', '"Fc_ref2" :')
                        parameters = parameters.replace('"Fc_ref" :', '"Fc_ref1" :')
                    else:
                        parameters = parameters.replace('"Fc" :', '"Fc_ref" :')
                    parameters += \
f'''
                    "Fc" : {Fc:.4e},
'''
            if not rxn_type:
                continue

            lhs_split = lhs.split('>')

            running_rates = running_rates | {f'<{lhs_split[0].split('<')[1]}>' : {'rxn_type' :rxn_type, 'parameters' : parameters, 'param': A if rxn_type=='ARRHENIUS' else Fc if rxn_type=='TROE' else 0.}}
            
            reactants = lhs_split[1].split('+')

            reactants_string = ''''''
            for term in reactants:
                split = term.split('*')
                if len(split) > 1:
                    stoich = split[0]
                    reactant = split[1].strip()
                else:
                    stoich = 1.0
                    reactant = split[0].strip()
                if len(reactant) > 0:
                    if (reactant in soa_species or (reactant[-1] in ['I','J','K'] and reactant[:-1] in soa_species)):
                        reactant = 'SOA'
                if reactant in soag_species and len(reactant) > 0:
                    reactant = 'SOAG'
                reactants_string += \
f'''
                        "{reactant}" : {{}},
'''
            products = rhs.split('+')
            for i, product in enumerate(products):
                split = product.split('-')
                if len(split) > 1:
                    products[i] = split[0]
                    products.insert(i, split[1])
            
            products_string = ''''''
            soa_accumulator = 0.
            soag_accumulator = 0.
            soa_flag = False
            soag_flag = False
            for term in products:
                split = term.split('*')
                if len(split) > 1:
                    stoich = split[0]
                    product = split[1].strip()
                else:
                    stoich = 1.0
                    product = split[0].strip()
                if len(product) > 0:
                    if (product in soa_species or (product[-1] in ['I','J','K'] and product[:-1] in soa_species)):
                        product = 'SOA'
                        soa_accumulator += float(stoich)
                        soa_flag = True
                        continue
                if product in soag_species and len(product) > 0:
                    product = 'SOAG'
                    soag_accumulator += float(stoich)
                    soag_flag = True
                    continue
                products_string += \
f'''
                        "{product}" : {{ "yield" : {stoich} }},
'''            
            if soa_flag:
                products_string += \
f'''
                        "SOA" : {{ "yield" : {soa_accumulator} }},
'''
            if soag_flag:
                products_string += \
f'''
                        "SOAG" : {{ "yield" : {soag_accumulator} }},
'''

            json += \
f'''                {{
                    "type" : "{rxn_type}",
                    {parameters}
                    "reactants" : {{
                        {reactants_string[:-2] + reactants_string[-1]}
                    }},
                    "products" : {{
                        {products_string[:-2] + products_string[-1]}
                    }}
                }},
'''
        if last_line:
            json = json[:-2] + json[-1]
            json += \
'''            ]
        }
    ]
}
'''
            break
    return(json)


file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/mech_cb6r5_ae7_aq.def')

mech = file.readlines()

file.close()

mech_json = to_json(mech)

print(mech_json)

with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/modified_mechanism.json','w') as json:
    json.write(mech_json)