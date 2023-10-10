import React, { useState } from 'react';
import './Upload.css'


function Upload({ onFileUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    } else {
      alert('Por favor, selecione um arquivo antes de fazer o upload.');
    }
  };

  return (
    <div className="container-upload">
      <div className="upload-box">
        <div className="content">
          <div className="upload-icon">
            <img
              className=""
              src="https://res.cloudinary.com/www-santhoshthomas-xyz/image/upload/v1620293451/RapTor/folder_1_ipacc2.png"
              alt=""
            />
          </div>

          <h5 className="text">Arraste seu arquivo (.xlsx ou .csv) aqui</h5>
          <p>ou</p>

          <div className="upload-btn-wrapper">
            <button className="btn" onClick={handleUpload}>
              Upload
            </button>
            <input
              type="file"
              name="myfile"
              onChange={handleFileChange}
              accept=".xlsx, .csv"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Upload;
