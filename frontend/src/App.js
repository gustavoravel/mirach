import React, { useState } from 'react';
import Instructions from "./components/Instructions/Instructions";
import Upload from "./components/Upload/Upload";
import Form from "./components/Form/Form";

function App() {
  const [columns, setColumns] = useState([]);

  const handleFileUpload = (file) => {
    // Faça o processamento do arquivo aqui para extrair as colunas
    // Suponha que você tenha uma função 'extractColumns' que faz isso
    const extractedColumns = extractColumns(file); // Substitua com sua lógica real
  }

  return (
    <div className="App">
      <Instructions />
      <Upload onFileUpload={handleFileUpload}/>
      {columns.length > 0 && <Form dataSetColumns={columns} />}
    </div>
  );
}

export default App;
