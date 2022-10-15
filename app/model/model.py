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
  
    base_elem['age'] = test.age
    # for k, e in test:
    #     aux = str(e)
    #     if aux.isnumeric():
    #         base_elem[k] = e
    #     else:
    #         key = k+'_'+e
    #         base_elem[key] = 1
    
    data = list(base_elem.values())
    res = model.predict([data])
    return classes[res[0]]