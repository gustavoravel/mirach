import React from 'react';
import './Instructions.css'

function Instructions() {
  return (
    <div className="instructions">
      <h1>Simple Forecast</h1>
      <h2>Instruções para Utilizar o Aplicativo</h2>
      <ol>
        <li>Insira seu arquivo .xlsx ou .csv:</li>
        <p>
          Clique no botão "Upload" ou arraste seu arquivo .xlsx ou .csv para a área designada.
        </p>
        <li>Escolha a categoria de previsão:</li>
        <p>
          Selecione a categoria do seu conjunto de dados a partir da lista de categorias disponíveis.
        </p>
        <li>Relacione as colunas com as variáveis de previsão:</li>
        <p>
          Associe as colunas do seu arquivo ao campo correspondente da categoria de previsão.
        </p>
        <li>Escolha o método de preenchimento de células vazias:</li>
        <p>
          Selecione como deseja lidar com valores ausentes ou células vazias no seu conjunto de dados.
        </p>
        <li>Escolha o algoritmo de previsão:</li>
        <p>
          Selecione o algoritmo ou método de previsão que deseja utilizar.
        </p>
        <li>Informe a periodicidade das datas:</li>
        <p>
          Indique a frequência (por exemplo, diária, mensal) das datas no seu conjunto de dados.
        </p>
        <li>Escolha o horizonte de previsão:</li>
        <p>
          Defina o período futuro que deseja prever com base nos seus dados históricos.
        </p>
      </ol>
      <p>
        Após seguir essas etapas, você estará pronto para iniciar a previsão de série temporal. Certifique-se de que todas as configurações estão corretas antes de prosseguir.
      </p>
      <p>
        ATENÇÃO: O objetivo desta aplicação não é fazer previsões corretas, pois trata-se de um projeto de portfolio.
      </p>
    </div>
  );
}

export default Instructions;
