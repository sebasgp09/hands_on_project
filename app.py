from flask import Flask, request, render_template
import pandas as pd
from pycaret.regression import load_model, predict_model

# 1. Inicializar la aplicación Flask
app = Flask(__name__)

# 2. Cargar el modelo que guardaste en el Paso 1
# Asegúrate de que el nombre coincida con el archivo .pkl generado
model = load_model('modelo_final')

# 3. Ruta para la página principal (el formulario)
@app.route('/')
def home():
    return render_template('home.html')

# 4. Ruta para procesar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados desde el formulario HTML
    # Los convertimos en un DataFrame porque PyCaret lo necesita así
    input_data = [x for x in request.form.values()]
    cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    data_unseen = pd.DataFrame([input_data], columns=cols)
    
    # Realizar la predicción
    prediction = predict_model(model, data=data_unseen)
    
    # El resultado de PyCaret suele estar en una columna llamada 'prediction_label'
    result = round(prediction.prediction_label[0], 2)

    return render_template('home.html', prediction_text=f'El costo estimado del seguro es: ${result}')

if __name__ == '__main__':
    app.run(debug=True)