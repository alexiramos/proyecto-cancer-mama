// src/components/private/StatsSummary.jsx
import React from "react";

const StatsSummary = () => {
  return (
    <div className="row g-4 mb-5">
      <div className="col-md-4">
        <div className="card stat-card shadow-sm border-0 h-100">
          <div className="card-body">
            <h6 className="card-subtitle mb-2 text-muted">
              Imágenes Procesadas
            </h6>
            <p className="card-title h1 fw-bold">1,250</p>
          </div>
        </div>
      </div>

      <div className="col-md-4">
        <div className="card stat-card shadow-sm border-0 h-100">
          <div className="card-body">
            <h6 className="card-subtitle mb-2 text-muted">
              Diagnósticos Confirmados
            </h6>
            <p className="card-title h1 fw-bold">320</p>
          </div>
        </div>
      </div>

      <div className="col-md-4">
        <div className="card stat-card shadow-sm border-0 h-100">
          <div className="card-body">
            <h6 className="card-subtitle mb-2 text-muted">
              Precisión del Modelo
            </h6>
            <p className="card-title h1 fw-bold">95%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsSummary;
