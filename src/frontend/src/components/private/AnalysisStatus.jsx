// src/components/private/AnalysisStatus.jsx
import React from "react";

const AnalysisStatus = () => {
  const analysisPlaceholder =
    "https://placehold.co/400x250/e2e8f0/64748b?text=An%C3%A1lisis+de+Imagen";

  return (
    <div className="text-center mb-5">
      <h2 className="h4 mb-3">Análisis en Curso</h2>

      <div className="analysis-placeholder mx-auto">
        <img
          src={analysisPlaceholder}
          alt="Análisis en curso"
          className="img-fluid rounded-3"
        />
      </div>

      <h3 className="h5 mt-4 mb-1">No hay análisis en curso</h3>
      <p className="text-muted">
        Comienza un nuevo análisis para ver el progreso aquí.
      </p>
    </div>
  );
};

export default AnalysisStatus;
