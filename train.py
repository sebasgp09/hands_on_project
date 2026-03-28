from pycaret.datasets import get_data
from pycaret.regression import *

# 1. Cargar datos de prueba
data = get_data('insurance')

# 2. Configurar el experimento
s = setup(data, target='charges', session_id=123, verbose=False)

# 3. Entrenar un modelo rápido
lr = create_model('lr')

# 4. GUARDAR EL MODELO
save_model(lr, 'modelo_final')
print("modelo_final.pkl ha sido creado.")