from pycaret.datasets import get_data
from pycaret.regression import *

# 1. Cargar datos
data = get_data('insurance')

# 2. Configurar el pipeline
s = setup(data, target='charges', session_id=123)

# 3. Entrenar el modelo (linear regression en este caso)
lr = create_model('lr')

# 4. Guardar el pipeline completo
save_model(lr, 'modelo_final')
