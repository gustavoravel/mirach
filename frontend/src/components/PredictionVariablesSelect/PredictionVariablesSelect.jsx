import React, { useState, useEffect } from 'react';

const categoryToVariables = {
  'Previsão de Demanda': [
    'item_id', 'timestamp', 'demand', 'location', 'price', 'promotion_applied', 'category', 'brand', 'color', 'genre'
  ],
  'Previsão de Força de Trabalho': [
    'workforce_type', 'timestamp', 'workforce_demand', 'wages', 'shift_length', 'location'
  ],
  'Previsão de Vendas, Lucros e Gastos': [
    'metric_name', 'timestamp', 'metric_value', 'category'
  ],
  'Previsão de Tráfego Web': [
    'item_id', 'timestamp', 'value', 'category'
  ],
  'Controle de Inventário': [
    'resource_name', 'timestamp', 'resource_value', 'category'
  ],
  'Alocação de Recursos e Equipamentos': [
    'item_id', 'timestamp', 'demand', 'price', 'category', 'brand', 'lead_time', 'order_cycle', 'safety_stock'
  ],
};

function PredictionVariablesSelect({ category, onVariablesChange }) {
  const [selectedVariables, setSelectedVariables] = useState([]);

  useEffect(() => {
    // Define as variáveis selecionadas com base na categoria escolhida
    setSelectedVariables(categoryToVariables[category] || []);
  }, [category]);

  const handleVariableChange = (e) => {
    // Atualiza as variáveis selecionadas
    const selected = Array.from(e.target.selectedOptions, (option) => option.value);
    setSelectedVariables(selected);
    onVariablesChange(selected);
  };

  return (
    <div>
      <label>Variáveis de Previsão:</label>
      <select
        className="form-select"
        multiple
        value={selectedVariables}
        onChange={handleVariableChange}
      >
        {selectedVariables.map((variable) => (
          <option key={variable} value={variable}>
            {variable}
          </option>
        ))}
      </select>
    </div>
  );
}

export default PredictionVariablesSelect;