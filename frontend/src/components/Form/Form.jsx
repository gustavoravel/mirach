import React, { useState } from 'react';
import PredictionVariablesSelect from '../PredictionVariablesSelect/PredictionVariablesSelect';

function Form({ onFormSubmit, dataSetColumns }) {
  const [category, setCategory] = useState('');
  const [predictionVariables, setPredictionVariables] = useState('');
  const [fillingMethod, setFillingMethod] = useState('');
  const [predictionAlgorithm, setPredictionAlgorithm] = useState('');
  const [dateFrequency, setDateFrequency] = useState('');
  const [forecastHorizon, setForecastHorizon] = useState('');

  const handleSubmit = () => {
    // Valide os campos e prepare os dados para envio
    const formData = {
      category,
      predictionVariables,
      fillingMethod,
      predictionAlgorithm,
      dateFrequency,
      forecastHorizon,
    };

    // Chame a função de retorno com os dados do formulário
    onFormSubmit(formData);
  };

  return (
    <div className="form-modal">
        <h2>Preencha as Informações de Previsão</h2>
        <label>Categoria de Previsão:</label>
        <select
            className="form-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}>
            <option>Previsão de Demanda</option>
            <option>Previsão de Força de Trabalho</option>
            <option>Previsão de Vendas, Lucros e Gastos</option>
            <option>Previsão de Tráfego Web</option>
            <option>Controle de Inventário</option>
            <option>Alocação de Recursos e Equipamentos</option>
        </select>


        <label>Variáveis de Previsão:</label>
        <h4>Selecione uma coluna do seu conjunto de dados (à direita) que representa uma variável de previsão (à esquerda)</h4>
        <div >
        {variables.map((variable) => (
            <label key={variable.name}>
            {variable.name}
            </label>
        ))}
        <div>
            <select
            name="columnName"
            value={predictionVariables} // Certifique-se de que 'predictionVariables' é a variável de estado correta
            onChange={(e) => setPredictionVariables(e.target.value)}
            >
            {columns.map((column) => (
                <option key={column.name} name="columnName" value={column.name}>
                {column.name}
                </option>
            ))}
            </select>
        </div>
        </div>


        <label>Método de Preenchimento de Células Vazias:</label>
        <input type="text" value={fillingMethod} onChange={(e) => setFillingMethod(e.target.value)} />

        <label>Algoritmo de Previsão:</label>
        <select
            value={predictionAlgorithm}
            onChange={(e) => setPredictionAlgorithm(e.target.value)}>
            <option>ETS</option>
            <option>ARIMA</option>
            <option disabled>Prophet</option>
            <option disabled>GARCH</option>
            <option disabled>Nonlinear</option>
            <option disabled>Multivariate TS</option>
            <option disabled>Regime Switching</option>
            <option disabled>State Space</option>
            <option disabled>Kelman Filtering</option>
            <option disabled>Deep Learning</option>
        </select>

        <label>As datas têm um intervalo de:</label>
        <div>
            <input
            type="number"
            min="1"
            value={dateFrequency} // Certifique-se de que 'dateFrequency' seja o estado correto
            onChange={(e) => setDateFrequency(e.target.value)}
            />
        </div>
        <div>
            <select>
                <option>hora(s)</option>
                <option>dia(s)</option>
                <option>semana(s)</option>
                <option>mes(es)</option>
                <option>bimestre(s)</option>
                <option>trimestre(s)</option>
                <option>semestre(s)</option>
                <option>ano(s)</option>
            </select>
        </div>

        <label>Fazer previsões para os próximos:</label>
        <div className="col-sm-2">
            <input
            type="number"
            min="1"
            max="10"
            value={forecastHorizon} // Certifique-se de que 'forecastHorizon' seja o estado correto
            onChange={(e) => setForecastHorizon(e.target.value)}
            />
        </div>
        <div className="col-sm-5">
            <label className="col-sm-2 col-form-label">períodos</label>
        </div>

      <button onClick={handleSubmit}>Iniciar Previsão</button>
    </div>
  );
}

export default Form;
