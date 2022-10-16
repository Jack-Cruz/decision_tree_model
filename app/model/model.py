import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_model-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

classes = [
    "No adherencia",
    "Adherencia"
]

def predict_pipeline(test):
    base_elem = {'age':0, 'medication':0, 'SAMS_item1':0, 'SAMS_item3':0, 'SAMS_item6':0,
       'SAMS_item10':0, 'SAMS_item11':0, 'SAMS_item15':0, 'SAMS_item16':0,
       'SAMS_item17':0, 'SAMS_item19':0, 'sex_hombre':0, 'sex_mujer':0,
       'marital status_casado':0, 'marital status_soltero':0,
       'marital status_viudo o divorciado':0, 'Education_primaria incompleta':0,
       'Education_secundaria completa':0, 'Education_secundaria incompleta':0,
       'Education_universitaria o tecnica completa':0,
       'Education_universitaria o tecnica incompleta':0,
       'Medication preparation by_sin vincular':0,
       'Medication preparation by_vinculado':0, 'Education_primaria completa':0,
       'Education_universitaria completa':0,
       'Education_universitaria incompleta':0}
    
    # Variables directas (tipo numérico)
    base_elem['age'] = test.age
    base_elem['medication'] = test.medication
    base_elem['SAMS_item1'] = test.SAMS_item1
    base_elem['SAMS_item3'] = test.SAMS_item3
    base_elem['SAMS_item6'] = test.SAMS_item6
    base_elem['SAMS_item10'] = test.SAMS_item10
    base_elem['SAMS_item11'] = test.SAMS_item11
    base_elem['SAMS_item15'] = test.SAMS_item15
    base_elem['SAMS_item17'] = test.SAMS_item17
    base_elem['SAMS_item19'] = test.SAMS_item19
    
    # Variable sex
    if test.sex == 'mujer':
        base_elem['sex_mujer'] = 1
    elif test.sex == 'hombre':
        base_elem['sex_hombre'] = 1

    # Variable marital_status
    if test.marital_status == 'casado':
        base_elem['marital status_casado'] = 1
    elif test.marital_status == 'soltero':
        base_elem['marital status_soltero'] = 1
    elif test.marital_status == 'viudo o divorciado':
        base_elem['marital status_viudo o divorciado'] = 1
    
    # Variable Education
    if test.Education == 'primaria completa':
        base_elem['Education_primaria completa'] = 1
    elif test.Education == 'primaria incompleta':
        base_elem['Education_primaria incompleta'] = 1
    elif test.Education == 'secundaria completa':
        base_elem['Education_secundaria completa'] = 1
    elif test.Education == 'secundaria incompleta':
        base_elem['Education_secundaria incompleta']
    elif test.Education == 'universitaria o tecnica completa':
        base_elem['Education_universitaria o tecnica completa'] = 1
    elif test.Education == 'universitaria o tecnica incompleta':
        base_elem['Education_universitaria o tecnica incompleta'] = 1
    elif test.Education == 'universitaria completa':
        base_elem['Education_universitaria completa'] = 1
    elif test.Education == 'universitaria incompleta':
        base_elem['Education_universitaria incompleta'] = 1
    
    # Variable Medication_preparation_by
    if test.Medication_preparation_by == 'sin vincular':
        base_elem['Medication preparation by_sin vincular'] = 1
    elif test.Medication_preparation_by == 'vinvulado':
        base_elem['Medication preparation by_vinculado'] = 1
    
    data = list(base_elem.values())
    res = model.predict([data])
    return classes[res[0]]